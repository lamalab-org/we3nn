import pytest
import torch
from e3nn import o3

from e3nn_WE import dihedral_group, planar_o3, restrict_o3, restricted_o3_couplings


def test_planar_embedding_and_l1_xy_z_decomposition():
    group = dihedral_group(6)
    embedding = planar_o3(group)
    assert embedding.check_embedding()
    restricted = restrict_o3(o3.Irrep("1o"), embedding)
    assert restricted.dim == 3 and restricted.check_representation()
    decomposition = restricted.decompose()
    assert sorted(irrep.id for irrep in decomposition.irreps) == [(0, 0), (1, 1)]


@pytest.mark.parametrize("degree", range(5))
def test_higher_l_d6_restrictions_decompose_and_reconstruct(degree):
    group = dihedral_group(6)
    restricted = restrict_o3(o3.Irrep(degree, (-1) ** degree), planar_o3(group))
    decomposition = restricted.decompose()
    assert sum(irrep.size for irrep in decomposition.irreps) == 2 * degree + 1
    for element in group.elements:
        torch.testing.assert_close(
            decomposition.reconstruct(element), restricted(element), atol=1e-10, rtol=1e-10
        )


def test_o3_coupling_is_inside_full_d6_intertwiner_space_and_space_can_be_larger():
    group = dihedral_group(6)
    embedding = planar_o3(group)
    left = restrict_o3(o3.Irrep("1o"), embedding)
    right = restrict_o3(o3.Irrep("1o"), embedding)
    output = restrict_o3(o3.Irrep("0e"), embedding)
    inherited = restricted_o3_couplings("1o", "1o", "0e")
    for element in group.elements:
        transformed = torch.einsum(
            "oa,pabc,ib,jc->poij",
            output(element), inherited, left(element), right(element),
        )
        torch.testing.assert_close(transformed, inherited, atol=3e-6, rtol=3e-6)

    from e3nn_WE import intertwiner_basis, tensor_product_representation

    full = intertwiner_basis(tensor_product_representation(left, right), output)
    assert full.shape[0] >= inherited.shape[0]

    # D6 admits more couplings for l=3 x l=3 -> l=2 than O(3)'s single path.
    l3 = restrict_o3(o3.Irrep("3o"), embedding)
    l2 = restrict_o3(o3.Irrep("2e"), embedding)
    finite = intertwiner_basis(tensor_product_representation(l3, l3), l2)
    assert finite.shape[0] > restricted_o3_couplings("3o", "3o", "2e").shape[0]
