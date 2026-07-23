"""
Jax Assertions: Nested Tree Utils vs jax.tree_util
==================================================

Verifies that ``autofit.graphical.utils.nested_*`` (autofit's recursive tree
walking utilities) agree with ``jax.tree_util`` as a reference implementation
across:

- ``nested_get`` / ``nested_set`` indexing into dict/tuple/list trees
- ``nested_zip`` and ``nested_filter`` ordering matches ``tree_flatten``
- ``nested_items`` paths match ``tree_flatten_with_path`` (after key
  normalization)
- ``nested_map`` / ``nested_iter`` propagating across heterogeneous trees
- ``nested_update`` semantics with NamedTuples preserved

The ``jax_*`` helpers below convert between jax's typed key path
(``SequenceKey`` / ``DictKey`` / ``GetAttrKey``) and autofit's plain tuple
key path so the two utility families can be compared directly.

Previously: ``test_autofit/graphical/functionality/test_nested.py``.
"""
# ENV: jax
# JAX assertion scripts test JAX behaviour; disabling JAX makes their
# assertions vacuous.

import collections

import jax.tree_util as tree_util

from autofit.graphical import utils

NTuple = collections.namedtuple("NTuple", "first, last")


def jax_nested_zip(tree, *rest):
    leaves, treedef = tree_util.tree_flatten(tree)
    return zip(leaves, *(treedef.flatten_up_to(r) for r in rest))


def jax_key_to_val(key):
    if isinstance(key, tree_util.SequenceKey):
        return key.idx
    elif isinstance(key, (tree_util.DictKey, tree_util.FlattenedIndexKey)):
        return key.key
    elif isinstance(key, tree_util.GetAttrKey):
        return key.name
    return key


def jax_path_to_key(path):
    return tuple(map(jax_key_to_val, path))


"""
__nested_get__
"""
obj = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}, "d": (3, {"e": [4, 5]})}

assert utils.nested_get(obj, ("b",)) == 2
assert utils.nested_get(obj, ("c", "a")) == 1
assert utils.nested_get(obj, ("d", 0)) == 3
assert utils.nested_get(obj, ("d", 1, "e", 1)) == 5

"""
__nested_set__
"""
obj = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}, "d": (3, {"e": [4, 5]})}

utils.nested_set(obj, ("b",), 3)
assert utils.nested_get(obj, ("b",))

utils.nested_set(obj, ("c", "a"), 2)
assert utils.nested_get(obj, ("c", "a")) == 2

utils.nested_set(obj, ("d", 1, "e", 1), 6)
assert utils.nested_get(obj, ("d", 1, "e", 1)) == 6

# Setting into an immutable tuple member must raise.
try:
    utils.nested_set(obj, ("d", 0), 4)
except TypeError:
    pass
else:
    raise AssertionError("nested_set into tuple should have raised TypeError")

"""
__nested_zip and nested_filter Ordering Matches tree_flatten__

Verify autofit's nested_zip walks the tree in the same order as jax's
tree_flatten, and that nested_filter agrees with raw equality, both for
plain dict/tuple/list trees AND trees containing NamedTuples.
"""
obj1 = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}, "d": (3, {"e": [4, 5]})}
obj2 = {"a": 1, "b": 2, "d": (3, {"e": [4, 5]}), "c": {"b": 2, "a": 1}}

assert all(v1 == v2 for (v1, v2) in utils.nested_zip(obj1, obj2))
assert all(utils.nested_filter(lambda x, y: x == y, obj1, obj2))
assert list(utils.nested_zip(obj1)) == list(utils.nested_zip(obj2))
assert list(utils.nested_zip(obj1, obj2)) == list(jax_nested_zip(obj1, obj2))

obj1 = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}, "d": (3, {"e": NTuple(4, 5)})}
obj2 = {"a": 1, "b": 2, "d": (3, {"e": NTuple(4, 5)}), "c": {"b": 2, "a": 1}}

assert all(v1 == v2 for (v1, v2) in utils.nested_zip(obj1, obj2))
assert all(utils.nested_filter(lambda x, y: x == y, obj1, obj2))
assert list(utils.nested_zip(obj1)) == list(utils.nested_zip(obj2))
assert list(utils.nested_zip(obj1, obj2)) == list(jax_nested_zip(obj1, obj2))

"""
__nested_items Paths Match tree_flatten_with_path__
"""
obj1 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, "c": (3, {"e": [4, 5]})}

for (k1, v1), (p2, v2) in zip(
    utils.nested_items(obj1), tree_util.tree_flatten_with_path(obj1)[0]
):
    assert k1 == jax_path_to_key(p2)
    assert v1 == v2

"""
__nested_filter By Predicate__
"""
obj1 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, "c": (3, {"e": [4, 5]})}
assert list(utils.nested_filter(lambda x: x % 2 == 0, obj1)) == [(2,), (4,), (2,)]

obj1 = {"b": 2, "a": 1, "c": (3, {"e": [4, 5]}), "d": {"b": 2, "a": 1}}
assert list(utils.nested_filter(lambda x: x % 2 == 0, obj1)) == [(2,), (4,), (2,)]

"""
__nested_map Across Heterogeneous Trees__
"""
obj1 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, "c": (3, {"e": [4, 5]})}
obj2 = {"a": 2, "b": 4, "c": (6, {"e": [8, 10]}), "d": {"a": 2, "b": 4}}
obj12 = utils.nested_map(lambda x: x * 2, obj1)
assert obj12 == obj2

obj3 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, "c": (3, {"e": (4, 5)})}
obj4 = {"a": 2, "b": 4, "c": (6, {"e": (8, 10)}), "d": {"a": 2, "b": 4}}
obj32 = utils.nested_map(lambda x: x * 2, obj3)
assert obj32 == obj4

obj5 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, "c": (3, {"e": NTuple(4, 5)})}
obj6 = {"a": 2, "b": 4, "c": (6, {"e": NTuple(8, 10)}), "d": {"a": 2, "b": 4}}
obj52 = utils.nested_map(lambda x: x * 2, obj5)
assert obj52 == obj6 == obj4

assert obj32 != obj2
assert obj52 != obj2

assert all(
    utils.nested_iter(utils.nested_map(lambda a, b, c: a == b == c, obj1, obj3, obj5))
)
assert all(
    utils.nested_iter(utils.nested_map(lambda a, b, c: a == b == c, obj2, obj4, obj6))
)
assert all(map(lambda x: x[0] == x[1] == x[2], utils.nested_zip(obj1, obj3, obj5)))
assert all(map(lambda x: x[0] == x[1] == x[2], utils.nested_zip(obj2, obj32, obj52)))

"""
__nested_update Preserves NamedTuple Type__
"""
assert utils.nested_update([1, (2, 3), [3, 2, {1, 2}]], {2: "a"}) == [
    1,
    ("a", 3),
    [3, "a", {1, "a"}],
]
assert utils.nested_update([1, NTuple(2, 3), [3, 2, {1, 2}]], {2: "a"}) == [
    1,
    ("a", 3),
    [3, "a", {1, "a"}],
]
assert isinstance(
    utils.nested_update([1, NTuple(2, 3), [3, 2, {1, 2}]], {2: "a"})[1], NTuple
)
assert utils.nested_update([{2: 2}], {2: "a"}) == [{2: "a"}]

"""
__nested_items Cross-Tree Lookup__

(Original test file had two functions named `test_nested_items` — pytest
ran both due to last-defined wins; we run the second variant since it's
the more comprehensive one.)
"""
obj1 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, "c": (3, {"e": [4, 5]})}
obj2 = {"a": 2, "b": 4, "c": (6, {"e": [8, 10]}), "d": {"a": 2, "b": 4}}
obj3 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, "c": (3, {"e": (4, 5)})}
obj4 = {"a": 2, "b": 4, "c": (6, {"e": (8, 10)}), "d": {"a": 2, "b": 4}}
obj5 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, "c": (3, {"e": NTuple(4, 5)})}
obj6 = {"a": 2, "b": 4, "c": (6, {"e": NTuple(8, 10)}), "d": {"a": 2, "b": 4}}

for path, val in utils.nested_items(obj1):
    assert (
        utils.nested_getitem(obj2, path) == utils.nested_getitem(obj4, path) == val * 2
    )

for path, val in utils.nested_items(obj3):
    assert (
        utils.nested_getitem(obj4, path) == utils.nested_getitem(obj6, path) == val * 2
    )

for path, val in utils.nested_items(obj5):
    assert (
        utils.nested_getitem(obj6, path) == utils.nested_getitem(obj2, path) == val * 2
    )

assert list(utils.nested_items([NTuple(1, 2), {2: 5, 1: 3}])) == [
    ((0, 0), 1),
    ((0, 1), 2),
    ((1, 1), 3),
    ((1, 2), 5),
]

assert list(utils.nested_items([1, (2, 3), [3, {"a": 1, "b": 2}]])) == list(
    utils.nested_items([1, (2, 3), [3, {"b": 2, "a": 1}]])
)
assert list(
    utils.nested_items(
        [
            1,
            (2, 3),
            [
                3,
                {
                    "b": 2,
                    "a": 1,
                },
            ],
        ]
    )
) == list(utils.nested_items([1, (2, 3), [3, {"b": 2, "a": 1}]]))

obj1 = [1, (2, 3), [3, {"b": 2, "a": 1}]]
obj2 = [1, (2, 3), [3, {"a": 1, "b": 2}]]
obj3 = [1, NTuple(2, 3), [3, {"a": 1, "b": 2}]]

if hasattr(tree_util, "tree_flatten_with_path"):
    jax_flat = tree_util.tree_flatten_with_path(obj1)[0]
    af_flat = utils.nested_items(obj2)

    for (jpath, jval), (akey, aval) in zip(jax_flat, af_flat):
        jkey = jax_path_to_key(jpath)
        assert jkey == akey
        assert jval == aval
        assert (
            utils.nested_get(obj2, jkey)
            == utils.nested_get(obj1, jkey)
            == utils.nested_get(obj2, akey)
            == utils.nested_get(obj1, akey)
        )

    jax_flat = tree_util.tree_flatten_with_path(obj2)[0]
    af_flat = utils.nested_items(obj3)
    for (jpath, jval), (akey, aval) in zip(jax_flat, af_flat):
        jkey = jax_path_to_key(jpath)
        assert jkey == akey
        assert jval == aval
        assert (
            utils.nested_get(obj2, jkey)
            == utils.nested_get(obj1, jkey)
            == utils.nested_get(obj2, akey)
            == utils.nested_get(obj1, akey)
            == utils.nested_get(obj3, jkey)
            == utils.nested_get(obj3, akey)
        )

print("nested: all assertions passed")
