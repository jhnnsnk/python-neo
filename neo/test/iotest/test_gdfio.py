# -*- coding: utf-8 -*-
"""
Tests of neo.io.exampleio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io.gdfio import GdfIO
from neo.test.iotest.common_io_test import BaseTestIO
import quantities as pq

class TestGdfIO(BaseTestIO, unittest.TestCase):
    ioclass = GdfIO
    files_to_test = ['gdf_test_files/spikes_default.gdf',
                     ]
    files_to_download = []

    # def test_read_segment_lazy(self):
    #     r = GdfIO(filename=None)
    #     seg = r.read_segment(cascade=True, lazy=True)
    #     for ana in seg.analogsignals:
    #         self.assertEqual(ana.size, 0)
    #         assert hasattr(ana, 'lazy_shape')
    #     for st in seg.spiketrains:
    #         self.assertEqual(st.size, 0)
    #         assert hasattr(st, 'lazy_shape')

    #     seg = r.read_segment(cascade=True, lazy=False)
    #     for ana in seg.analogsignals:
    #         self.assertNotEqual(ana.size, 0)
    #     for st in seg.spiketrains:
    #         self.assertNotEqual(st.size, 0)

    # def test_read_segment_cascade(self):
    #     r = GdfIO(filename=None)
    #     seg = r.read_segment(cascade=False)
    #     self.assertEqual(len(seg.analogsignals), 0)
    #     seg = r.read_segment(cascade=True, num_analogsignal=4)
    #     self.assertEqual(len(seg.analogsignals), 4)

    # def test_read_analogsignal(self):
    #     r = GdfIO(filename=None)
    #     r.read_analogsignal(lazy=False, segment_duration=15., t_start=-1)

    def test_read_spiketrain(self):
        for fname in self.files_to_test :
            r = GdfIO(filename=fname)
            r.read_spiketrain(gdf_id=1, t_stop=1000000., lazy=False)
            r.read_segment(gdf_id_list=[1], t_stop=1000000., lazy=False)

        r = GdfIO(filename='gdf_test_files/withgidF-time_in_stepsF-1254-0.gdf')
        r.read_spiketrain(t_stop=1000000., lazy=False, id_column=None, time_column=0)
        r.read_segment(t_stop=1000000., lazy=False, id_column=None, time_column=0)

        r = GdfIO(filename='gdf_test_files/withgidF-time_in_stepsT-1256-0.gdf')
        st = r.read_spiketrain(t_stop=1000000., time_unit=pq.CompoundUnit('0.1*ms'), lazy=False, id_column=None, time_column=0)
        st = r.read_segment(t_stop=1000000., time_unit=pq.CompoundUnit('0.1*ms'), lazy=False, id_column=None, time_column=0)

        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsF-1255-0.gdf')
        r.read_spiketrain(gdf_id=1, t_stop=1000000., lazy=False, id_column=0, time_column=1)
        r.read_segment(gdf_id_list=[1], t_stop=1000000., lazy=False, id_column=0, time_column=1)

        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsT-1257-0.gdf')
        r.read_spiketrain(gdf_id=1, t_stop=1000000., lazy=False, id_column=0, time_column=1)
        r.read_segment(gdf_id_list=[1], t_stop=1000000., lazy=False, id_column=0, time_column=1)


if __name__ == "__main__":
    unittest.main()
