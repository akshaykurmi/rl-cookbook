import numpy as np
import pytest

from rl.utils import RingBuffer


class TestRingBuffer:

    @staticmethod
    def _create_buffer(size, shape, dtype, fill):
        buffer = RingBuffer(size, shape, dtype)
        for i in range(fill):
            buffer.append(i)
        return buffer

    @pytest.fixture
    def empty(self):
        return self._create_buffer(10, (), np.float32, 0)

    @pytest.fixture
    def partially_full(self):
        return self._create_buffer(10, (), np.float32, 5)

    @pytest.fixture
    def full(self):
        return self._create_buffer(10, (), np.float32, 10)

    @pytest.fixture
    def overflown(self):
        return self._create_buffer(10, (), np.float32, 15)

    @pytest.fixture
    def empty_expected(self):
        return np.array([], dtype=np.float32)

    @pytest.fixture
    def partially_full_expected(self):
        return np.arange(5, dtype=np.float32)

    @pytest.fixture
    def full_expected(self):
        return np.arange(10, dtype=np.float32)

    @pytest.fixture
    def overflown_expected(self):
        return np.arange(5, 15, dtype=np.float32)

    class TestPurge:

        def test_purge_empties_the_buffer(self, empty, partially_full, full, overflown):
            for buffer in [empty, partially_full, full, overflown]:
                buffer.purge()
                assert len(buffer) == 0
                for i in range(-10, 10):
                    with pytest.raises(IndexError): buffer[i]
                    with pytest.raises(IndexError): buffer[-i]

    class TestAppend:

        def test_append_adds_values_to_end_of_buffer(self):
            buffer = RingBuffer(10, (), np.int32)
            for i in range(10):
                buffer.append(i)
                assert np.array_equal(buffer[:], np.arange(i + 1))

        def test_append_overwrites_in_fifo_order(self):
            buffer = RingBuffer(3, (), np.int32)
            for i in range(10):
                buffer.append(i)
                assert np.array_equal(buffer[:], np.arange(max(0, i - 2), i + 1))

    class TestGetItem:

        def test_invalid_index(self, full):
            with pytest.raises(IndexError): full[1:3, 5:8]
            with pytest.raises(IndexError): full[4.99]
            with pytest.raises(IndexError): full[3.45:9]
            with pytest.raises(IndexError): full['xyz']

        def test_valid_int_indices(self, partially_full, full, overflown,
                                   partially_full_expected, full_expected, overflown_expected):
            for i in range(-5, 5): assert partially_full[i] == partially_full_expected[i]
            for i in range(-10, 10): assert full[i] == full_expected[i]
            for i in range(-10, 10): assert overflown[i] == overflown_expected[i]

        def test_out_of_bounds_int_indices(self, empty, partially_full, full, overflown):
            for i in range(10):
                with pytest.raises(IndexError): empty[i]
                with pytest.raises(IndexError): empty[-i]
                with pytest.raises(IndexError): partially_full[5 + i]
                with pytest.raises(IndexError): partially_full[-6 - i]
                with pytest.raises(IndexError): full[10 + i]
                with pytest.raises(IndexError): full[-11 - i]
                with pytest.raises(IndexError): overflown[10 + i]
                with pytest.raises(IndexError): overflown[-11 - i]

        def test_slices(self, empty, partially_full, full, overflown,
                        empty_expected, partially_full_expected, full_expected, overflown_expected):
            for s in [None, *range(1, 15)]:
                for i in [None, *range(-15, 15)]:
                    for j in [None, *range(-15, 15)]:
                        assert np.array_equal(empty[i:j:s], empty_expected[i:j:s])
                        assert np.array_equal(partially_full[i:j:s], partially_full_expected[i:j:s])
                        assert np.array_equal(full[i:j:s], full_expected[i:j:s])
                        assert np.array_equal(overflown[i:j:s], overflown_expected[i:j:s])

    class TestSetItem:

        @pytest.fixture
        def check_assign(self):
            def func(actual, expected, key):
                v1 = np.random.uniform()
                v2 = np.random.uniform(size=(1 if isinstance(key, int) else len(expected[key])))
                for v in [v1, v2]:
                    actual[key] = v
                    expected[key] = v
                    assert np.array_equal(actual, expected)

            return func

        def test_invalid_index(self, full):
            with pytest.raises(IndexError): full[1:3, 5:8] = np.random.uniform()
            with pytest.raises(IndexError): full[4.99] = np.random.uniform()
            with pytest.raises(IndexError): full[3.45:9] = np.random.uniform()
            with pytest.raises(IndexError): full['xyz'] = np.random.uniform()

        def test_invalid_value_shape(self, overflown):
            with pytest.raises(ValueError): overflown[3] = []
            with pytest.raises(ValueError): overflown[3] = [1, 2, 3]
            with pytest.raises(ValueError): overflown[2:5] = [1, 2]
            with pytest.raises(ValueError): overflown[2:5] = [1, 2, 3, 4]
            with pytest.raises(ValueError): overflown[2:5:2] = [1, 2, 3]

        def test_valid_int_indices(self, partially_full, full, overflown, check_assign,
                                   partially_full_expected, full_expected, overflown_expected):
            for i in range(-5, 5): check_assign(partially_full, partially_full_expected, i)
            for i in range(-10, 10): check_assign(full, full_expected, i)
            for i in range(-10, 10): check_assign(overflown, overflown_expected, i)

        def test_out_of_bounds_int_indices(self, empty, partially_full, full, overflown):
            for i in range(10):
                with pytest.raises(IndexError): empty[i] = np.random.uniform()
                with pytest.raises(IndexError): empty[-i] = np.random.uniform()
                with pytest.raises(IndexError): partially_full[5 + i] = np.random.uniform()
                with pytest.raises(IndexError): partially_full[-6 - i] = np.random.uniform()
                with pytest.raises(IndexError): full[10 + i] = np.random.uniform()
                with pytest.raises(IndexError): full[-11 - i] = np.random.uniform()
                with pytest.raises(IndexError): overflown[10 + i] = np.random.uniform()
                with pytest.raises(IndexError): overflown[-11 - i] = np.random.uniform()

        def test_slices(self, check_assign, empty, partially_full, full, overflown,
                        empty_expected, partially_full_expected, full_expected, overflown_expected):
            for s in [None, *range(1, 15)]:
                for i in [None, *range(-15, 15)]:
                    for j in [None, *range(-15, 15)]:
                        check_assign(empty, empty_expected, slice(i, j, s))
                        check_assign(partially_full, partially_full_expected, slice(i, j, s))
                        check_assign(full, full_expected, slice(i, j, s))
                        check_assign(overflown, overflown_expected, slice(i, j, s))
