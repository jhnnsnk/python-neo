# -*- coding: utf-8 -*-
"""
Tests of neo.io.blackrockio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import os
import struct
import sys
import tempfile

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo import NeuralynxIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io import tools
from neo.test.tools import assert_arrays_almost_equal
from neo.core import Segment


#~ class testRead(unittest.TestCase):
    #~ """Tests that data can be read from KlustaKwik files"""
    #~ def test1(self):
        #~ """Tests that data and metadata are read correctly"""
        #~ pass
    #~ def test2(self):
        #~ """Checks that cluster id autosets to 0 without clu file"""
        #~ pass
        #~ dirname = os.path.normpath('./files_for_tests/klustakwik/test2')
        #~ kio = neo.io.KlustaKwikIO(filename=os.path.join(dirname, 'base2'),
            #~ sampling_rate=1000.)
        #~ block = kio.read()
        #~ seg = block.segments[0]
        #~ self.assertEqual(len(seg.spiketrains), 1)
        #~ self.assertEqual(seg.spiketrains[0].name, 'unit 0 from group 5')
        #~ self.assertEqual(seg.spiketrains[0].annotations['cluster'], 0)
        #~ self.assertEqual(seg.spiketrains[0].annotations['group'], 5)
        #~ self.assertEqual(seg.spiketrains[0].t_start, 0.0)
        #~ self.assertTrue(np.all(seg.spiketrains[0].times == np.array(
            #~ [0.026, 0.122, 0.228])))


class CommonTests(BaseTestIO, unittest.TestCase):
    ioclass = NeuralynxIO
    read_and_write_is_bijective = False
    hash_conserved_when_write_read = False

    files_to_test = [
        #'test2/test.ns5'
        ]

    files_to_download = [
        #'test2/test.ns5'
        ]

    local_test_dir = None


@unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
class testRead(unittest.TestCase):
    def setUp(self):
        self.sn = ('/home/sprenger/Documents/projects/SPPHamburg/Data/'
                    'causmech/2014-07-24_10-29-07')
#        self.sn = os.path.join(tempfile.gettempdir(),
#                               'files_for_testing_neo',
#                               'blackrock/test2/test.ns5')
        if not os.path.exists(self.sn):
            raise unittest.SkipTest('data file does not exist:' + self.sn)

    def test_read_block_in_range(self):
        """Read data in a certain time range into one block"""
        nio = NeuralynxIO(self.sn)
        block = nio.read_block(t_starts=[0.1*pq.s], t_stops=[0.2*pq.s])
        self.assertEqual(len(nio.parameters_ncs), 16)
        self.assertTrue(nio.parameters_nev['digital_markers']
                                            == {11: 'Starting Recording'})

        # Everything put in one segment
        self.assertEqual(len(block.segments), 1)
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignalarrays), 16)

        self.assertTrue(all([(seg.analogsignalarrays[i].sampling_rate.units ==
                                pq.CompoundUnit('32*kHz')) for i in range(16)]))
        self.assertTrue(all([seg.analogsignalarrays[i].t_start == 0.1*pq.s
                                for i in range(16)]))
        self.assertTrue(all([seg.analogsignalarrays[i].t_stop == 0.2*pq.s
                                for i in range(16)]))

    def test_read_segment(self):
        """Read data into one segment"""
        nio = NeuralynxIO(self.sn)
        seg = nio.read_segment(channel_list=[1,2,3])
        self.assertEqual(len(nio.parameters_ncs), 16)
        self.assertTrue(nio.parameters_nev['digital_markers']
                                            == {11: 'Starting Recording'})

        self.assertEqual(len(seg.analogsignalarrays), 3)

        self.assertTrue(all([(seg.analogsignalarrays[i].sampling_rate.units ==
                                pq.CompoundUnit('32*kHz')) for i in range(3)]))
        self.assertTrue(all([seg.analogsignalarrays[i].t_start == 0*pq.s
                                for i in range(3)]))
        self.assertTrue(all([(seg.analogsignalarrays[i].t_stop
                                == 40448 * pq.CompoundUnit('1/(32000*Hz)'))
                                for i in range(3)]))


    def test_read_analogsignalarray(self):
        """Read data into one segment"""
        nio = NeuralynxIO(self.sn)
        seg = Segment()
        nio.read_ncs('Tet4a',seg)
        anasig = seg.analogsignalarrays[0]

        self.assertTrue(anasig.sampling_rate.units == pq.CompoundUnit('32*kHz'))
        self.assertTrue(anasig.t_start == 0*pq.s for i in range(3))
        self.assertTrue(anasig.t_stop == 40448 * pq.CompoundUnit('1/(32000*Hz)'))



if __name__ == '__main__':
    unittest.main()
