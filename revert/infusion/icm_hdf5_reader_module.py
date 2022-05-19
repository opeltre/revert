#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 09:17:28 2021
Modified by Michal M. Placek

@author: ertabeqiri
"""
 # pylint: disable=trailing-whitespace
 # pylint: disable=trailing-newlines
 # pylint: disable=line-too-long
 # pylint: disable=too-few-public-methods



import numpy as np
import math

## assign fixed variables
INVALID_VALUE = -99999
INDEX_TABLE_INDEX = 0
INDEX_TABLE_TIME = 1
INDEX_TABLE_LENGTH = 2
INDEX_TABLE_FRQ = 3
QUALITY_TABLE_TIME = 0
QUALITY_TABLE_VALUE = 1
PREV_IF_IN_GAP = True
NEXT_IF_IN_GAP = False
MICROSEC_IN_SEC = 1000000

class Iterator:
    '''returns data stream of the next good continuous data section'''
    def __init__(self,reader):
        if not isinstance(reader, MyHdF5signalReaderClass):
            raise TypeError
        self._reader = reader
        self._value_index = reader.get_loaded_data_start_index()
        #self._sect_count = 0 # used for debugging only

    def __next__(self):

        # this function will return the start index and
        # the length of the next continuous data stream
        while not self._is_end_of_data():


            cont_section_start_index = self._find_cont_section_start_index()
            #print('iterator.cont_section_start_index = ' +  str(cont_section_start_index))

            self._value_index = cont_section_start_index

            # if section found
            #if (self._sect_count < 10) and (cont_section_start_index >= 0):

            end_index_plus1 = self._reader.get_loaded_data_start_index() + self._reader.get_loaded_data_length()
            if cont_section_start_index >= 0 and cont_section_start_index < end_index_plus1:

                cont_sect_end_index = self._find_cont_section_end_index()

                #print('iterator.cont_sect_end_index = ' +  str(cont_sect_end_index))
                #self._sect_count += 1

                if cont_sect_end_index >= 0:
                    if cont_sect_end_index > self._reader.get_loaded_data_end_index():
                        cont_sect_end_index = self._reader.get_loaded_data_end_index()

                    cont_sect_length = cont_sect_end_index - cont_section_start_index + 1
                    next_section = self._create_data_stream_obj(cont_section_start_index, cont_sect_length)

                    # move the current index past the end of the currently return section
                    self._value_index = cont_sect_end_index + 1

                    return next_section

                self._value_index = self._reader.find_next_normal_quality_section_index(self._value_index)

        raise StopIteration

    def _is_end_of_data(self):
        return self._value_index > self._reader.get_loaded_data_end_index()

    def _find_cont_section_start_index(self):
        # Check if the curent stream pointer index is within the normal data section
        is_normal = self._reader.is_normal_quality_point(self._value_index)

        # Find the end of the continuous valid section (taking into account quality table)
        if is_normal:
            return self._value_index
        return self._reader.find_next_normal_quality_section_index(self._value_index)

    def _find_cont_section_end_index(self):
        is_normal = self._reader.is_normal_quality_point(self._value_index)

        #print('_find_cont_section_end_index_is_normal = ' + str(self._reader.is_normal_quality_point(self._value_index)))
        if is_normal:
            cont_sect_end_index = self._reader.find_cont_section_end_index(self._value_index)
            normal_qual_end_index = self._reader.find_end_index_of_current_quality_section(self._value_index)
            #print( 'cont_idx=' + str(cont_sect_end_index) + ' normal_idx=' + str(normal_qual_end_index))
            if normal_qual_end_index < cont_sect_end_index:
                return normal_qual_end_index
            return cont_sect_end_index
        return -1

    def _create_data_stream_obj(self, cont_section_start_index, cont_sect_length):
        next_section = DataStreamClass()

        # Extract data values and store them in DataStream
        next_section.values = self._reader.get_section_values(cont_section_start_index, cont_sect_length)
        next_section.sampling_frq = self._reader.get_sampling_frq(cont_section_start_index)
        next_section.start_time_microsec = self._reader.index_to_time(cont_section_start_index)
        #print('Iterator._create_data_stream_obj.next_start_time =' + str(next_section.start_time_microsec) )

        return next_section


class MyHdF5signalReaderClass:
    '''reads the data file '''
    def __init__(self, hdf5_data):
        self._hdf5_data = hdf5_data
        self._sig_name = ''
        self._sig_stream = np.empty(shape=(0))
        self._index_tbl = np.empty(shape=(0,0))
        self._qual_tbl = np.empty(shape=(0,0))
 #       self._loaded_data_start_time = 0 # nothing is loaded
 #       self._loaded_data_end_time = 0 # nothing is loaded
        self._sampling_freq = 0 # nothing is loaded
        self._waves = None
        self._loaded_data_start_index = 0
        self._loaded_data_length = 0
        self.is_mock = False
        self.mock_data_array = np.empty(0)

    # Getter functions
#    def get_loaded_data_start_time(self):
#        return self._loaded_data_start_time

#    def get_loaded_data_end_time(self):
#        return self._loaded_data_end_time

    def get_sampling_freq(self):
        return self._sampling_freq

    def get_loaded_data_start_index(self):
        return self._loaded_data_start_index

    def get_loaded_data_length(self):
        return self._loaded_data_length

    def get_loaded_data_end_index(self):
        return self._loaded_data_start_index + self._loaded_data_length - 1

    def get_all_data_start_time(self):
        return self._index_tbl[0][INDEX_TABLE_TIME]

    def get_all_data_end_time(self):
        last_idx_entry = self._index_tbl[len(self._index_tbl)-1]
        tm = last_idx_entry[INDEX_TABLE_TIME]
        frq = last_idx_entry[INDEX_TABLE_FRQ]
        l = last_idx_entry[INDEX_TABLE_LENGTH]
        return tm + l / frq * MICROSEC_IN_SEC


    # initialize data stream
    def init_wave_data(self, signal_name, sig_group_name='waves'):
        # sig_group_name: 'waves' or 'numerics'
        self._sig_name = signal_name
        self._waves = self._hdf5_data[sig_group_name]

        ## read index table
        self._index_tbl = np.array(self._waves[self._sig_name + '.index'])

        ## extract data stream sampling frequency ( assumes that all index table entries have the same sampling frequency as it comes from ICM+ packadging!!!)
        self._sampling_freq = self._index_tbl[0][INDEX_TABLE_FRQ]

        ## read quality table
        self._qual_tbl = np.array(self._waves[self._sig_name + '.quality'])



    def _load_dataPortion_mock(self):
        #return np.array([99.,100,101,90,91,92,93,80,81,82,83,70,71,72])[self._loaded_data_start_index : self._loaded_data_end_index + 1]
        end_index = self._loaded_data_start_index + self._loaded_data_length
        return self.mock_data_array[self._loaded_data_start_index : end_index ]

    def _load_data_portion(self):
        #return np.array(self._waves[self._sig_name])[self._loaded_data_start_index : self._loaded_data_end_index + 1]
        end_index = self._loaded_data_start_index + self._loaded_data_length
        return np.array(self._waves[self._sig_name])[self._loaded_data_start_index : end_index ]

    # original raw data loader
    def load_raw_data_set(self,page_start_time, page_len_microsec):
        self._loaded_data_length = 0

        if page_len_microsec == 0:
            return

        ## calculate start index and end index
        start_val_index, start_delta_t = self.time_to_sample_index(page_start_time)

        end_val_index, end_delta_t = self.time_to_sample_index(page_start_time + page_len_microsec )


        if (start_val_index == self.get_last_value_index()) and end_val_index == self.get_last_value_index():
            # page starts at the last value and ends at the last value
            end_val_index = end_val_index
            start_val_index = start_val_index

        else:
            if end_delta_t == 0:
               # be exclusive for the last index
                end_val_index -= 1

            if end_val_index == -1:
                # page ends before data start
                return

            if (start_val_index == self.get_last_value_index()) and  (end_delta_t > 0):
                # page starts after the data ends
                return

            if (start_val_index == end_val_index) and (start_delta_t > 0):
                # page entirely in a gap
                return


        if start_delta_t == 0:
            self._loaded_data_start_index = start_val_index
        else:
            self._loaded_data_start_index = start_val_index + 1

        self._loaded_data_length = end_val_index - self._loaded_data_start_index + 1

        #print( 'reader load_raw_data_set.self._loaded_data_start_index ' + str(self._loaded_data_start_index))
        #print( 'reader load_raw_data_set.self._loaded_data_length ' + str(self._loaded_data_length ))

        ## read the signal stream
        if self.is_mock:
            self._sig_stream = self._load_dataPortion_mock()
        else:
            self._sig_stream = self._load_data_portion()

        ## replace -99999 ( = INVALID_VALUE) in stream with NAN
        self._sig_stream[self._sig_stream==INVALID_VALUE] = np.NaN

    def is_empty(self):
        return self._loaded_data_length == 0

    def is_empty_or_abnormal(self):
        '''returns true if the loaded data are empty or all non good quality'''

        if self.is_empty():
            return True

        if self.is_normal_quality_point(self._loaded_data_start_index):
            return False

        next_normal_qual_index = self.find_next_normal_quality_section_index(self._loaded_data_start_index)
        return (next_normal_qual_index == -1) or (next_normal_qual_index > self.get_loaded_data_end_index())


    def __iter__(self):
        return Iterator(self)

    def time_to_sample_index( self,time_stamp_microsec ):
        '''returns index of the data point corresponding to the time stamp.
        If the time stamp falls in the data gap it will use use_previous_if_in_gap
        to return the first sample index before the time stamp, or the first after'''

        ## takes care of time stamp falling before the first data value
        if time_stamp_microsec < self._index_tbl[0][INDEX_TABLE_TIME]:
            return -1, time_stamp_microsec - self._index_tbl[0][INDEX_TABLE_TIME]

        prev_value_idx = self.get_last_value_index()
        for i in range (0,len(self._index_tbl)):
            start_time_microsec = self._index_tbl[i][INDEX_TABLE_TIME]
            start_index = self._index_tbl[i][INDEX_TABLE_INDEX]
            sampling_frq = self._index_tbl[i][INDEX_TABLE_FRQ]
            length = self._index_tbl[i][INDEX_TABLE_LENGTH]
            end_time_microsec =  start_time_microsec + length / sampling_frq * MICROSEC_IN_SEC

            if time_stamp_microsec < start_time_microsec:
                prev_entry_start_idx = self._index_tbl[i-1][INDEX_TABLE_INDEX]
                prev_entry_len = self._index_tbl[i-1][INDEX_TABLE_LENGTH]
                prev_value_idx = prev_entry_start_idx + prev_entry_len  - 1
                break
                #return prev_value_idx, time_stamp_microsec - self.index_to_time(prev_value_idx)

            #if (start_time_microsec <= time_stamp_microsec) and (time_stamp_microsec <= end_time_microsec):
            if start_time_microsec <= time_stamp_microsec < end_time_microsec:
                index_float = start_index + (time_stamp_microsec - start_time_microsec) / MICROSEC_IN_SEC * sampling_frq
                #if use_previous_if_in_gap:
                #    return math.trunc( index_float )
                prev_value_idx = math.floor( index_float )
                break
                #return prev_value_idx, (index_float - prev_value_idx) / sampling_frq * MICROSEC_IN_SEC

#            if (i == len(self._index_tbl) - 1) and (time_stamp_microsec >= end_time_microsec):
         # prev_value_idx = self.get_last_value_index()
        return prev_value_idx, time_stamp_microsec - self.index_to_time(prev_value_idx)


    def get_last_value_index(self): ## is inclusive
        last_idx_entry_idx = len(self._index_tbl) - 1
        last_idx_entry = self._index_tbl[last_idx_entry_idx]
        last_val_index = last_idx_entry[INDEX_TABLE_INDEX] + last_idx_entry[INDEX_TABLE_LENGTH] - 1
        return last_val_index

    def index_to_time( self, value_index ):
        '''returns the time stamp of the data point corresponding to the index.'''
        #First calculates the time stamp of the start of the section, then adds the relative delta time

        index_table_entry_index = self.get_index_tbl_entry_index( value_index )
        index_tbl_idx = self._index_tbl[index_table_entry_index][INDEX_TABLE_INDEX]
        index_tbl_start_time_microsec = self._index_tbl[index_table_entry_index][INDEX_TABLE_TIME]
        sampling_frq = self._index_tbl[index_table_entry_index][INDEX_TABLE_FRQ]
        return index_tbl_start_time_microsec + (value_index - index_tbl_idx )/sampling_frq * MICROSEC_IN_SEC

    def get_section_values(self, start_value_index,length):
        '''returns signal values given the starting index and the length of the section'''
        page_value_index = start_value_index - self._loaded_data_start_index
        return self._sig_stream[int(page_value_index):int(page_value_index+length)]

    def get_sampling_frq(self, value_index):
        '''returns the sampling frequency of the data point corresponding to the index'''
        index_table_entry_index = self.get_index_tbl_entry_index( value_index )
        return self._index_tbl[index_table_entry_index][INDEX_TABLE_FRQ]

    def is_normal_quality_point( self, value_index ):
        '''checks if the data point corresponding to the index is in a good quality section.
        If so, the condition is True.'''
        qual_index = self.get_quality_entry_index( value_index )
        if qual_index >= 0:
            return self._qual_tbl[qual_index][QUALITY_TABLE_VALUE] == 0

        return False

    def get_quality_entry_index( self, value_index ):
        '''returns the quality table index related to the data point corresponding to the value index.'''
        #First calculates the time stamp of the value index, then iterates through 'Time' in quality table.
        #When it finds Time > time stamp, then assigns the index of the previous entry.

        time_stamp_microsec = self.index_to_time( value_index)
        if time_stamp_microsec < self._qual_tbl[0][QUALITY_TABLE_TIME]:
            return -1

        for i in range (1,len(self._qual_tbl)):
            if self._qual_tbl[i][QUALITY_TABLE_TIME] > time_stamp_microsec:
                return i-1
        return len(self._qual_tbl) - 1

    def get_quality_section_start_time(self, value_index):
        return self._qual_tbl[value_index][QUALITY_TABLE_TIME]


    def get_index_tbl_entry_index( self, value_index ):
        '''returns the index of the index table related to the data point that corresponds to the value index'''

        for i in range (1,len(self._index_tbl)):
            if self._index_tbl[i][INDEX_TABLE_INDEX] > (value_index):
                return i-1
        return len(self._index_tbl) - 1


    def find_cont_section_end_index( self, value_index ):
        '''calculates the index corresponding to the end of the continuous stream section of the data point corresponding to the value index '''
        current_index_entry_index = self.get_index_tbl_entry_index(value_index)
        return self._index_tbl[current_index_entry_index][INDEX_TABLE_INDEX] + self._index_tbl[current_index_entry_index][INDEX_TABLE_LENGTH] - 1


    def find_end_index_of_current_quality_section(self, value_index):
        '''calculates the data index corresponding to the end of the quality section of the data point corresponding to the current index'''
        qual_entry_index = self.get_quality_entry_index( value_index )

        if qual_entry_index == len(self._qual_tbl) - 1:
            return self.get_last_value_index()  # len(self._sig_stream) - 1

        next_qual_entry_time_stamp_microsec = self._qual_tbl[ qual_entry_index + 1][QUALITY_TABLE_TIME]

        sample_index, delta_t = self.time_to_sample_index( next_qual_entry_time_stamp_microsec)
        if delta_t == 0:
            sample_index -= 1
        return sample_index


    def find_next_normal_quality_section_index(self, value_index):
        '''returns the index of the data point in the next normal quality section.
        If the value index falls in a gap, it returns the index of the next continuous section '''
        cur_qual_index = self.get_quality_entry_index( value_index )

        if (cur_qual_index == -1) or ( self._qual_tbl[cur_qual_index][QUALITY_TABLE_VALUE] == 0 ):
            cur_qual_index += 1

        for i in range (cur_qual_index,len(self._qual_tbl)):
            if self._qual_tbl[i][QUALITY_TABLE_VALUE] == 0:
                normal_quality_start_idx, delta_t = self.time_to_sample_index( self._qual_tbl[i][QUALITY_TABLE_TIME])
                if delta_t > 0:
                    normal_quality_start_idx += 1
                return normal_quality_start_idx

        return self.get_last_value_index() + 1 #  get_data_length()


class DataStreamClass:
    '''stores the data values, the sampling frequency and start time of a data stream'''
    def __init__(self):
        self.values = np.empty( shape=(0))
        self.sampling_frq = 0
        self.start_time_microsec = -1


    def get_end_time_microsec(self):
        '''returns the the stamp of the end of the values array'''
        return self.start_time_microsec  + len(self.values) / self.sampling_frq * MICROSEC_IN_SEC

    def is_empty(self):
        '''checks if the values array is empty. If so, returns True'''
        return len(self.values) == 0

    def append_new_data_section(self, data_sec):
        '''appends a new section to the output buffer.
        If there was a gap after the previous data section, it will fill it in with NAN values.'''

        # data_sec represents a continuous good quality data section
        if self.sampling_frq == 0:
            self.sampling_frq = data_sec.sampling_frq

        if self.start_time_microsec < 0:
            self.start_time_microsec = data_sec.start_time_microsec - len(self.values) / self.sampling_frq * MICROSEC_IN_SEC

        if not self.is_empty():
            current_end_time = self.get_end_time_microsec()
            #print('DataStreamClass.append_new_data_section.current_end_time = '+ str(current_end_time))
            #print('DataStreamClass.append_new_data_section.start_time_microsec  = '+ str(data_sec.start_time_microsec))

            gap_data = get_gap_filler(data_sec.start_time_microsec - current_end_time, self.sampling_frq)
            self.values = np.append(self.values , gap_data)

        self.values = np.append(self.values , data_sec.values)

    def append_NAN_section(self,size):
        if size <= 0:
            return
        NAN_section = np.empty(size)
        NAN_section[:] = np.NaN
        self.values = np.append(self.values , NAN_section)


# Utility functions
def get_gap_filler(duration_in_microsec,sampling_frq):
    '''creates an array of empty values'''
    length = round(duration_in_microsec / MICROSEC_IN_SEC * sampling_frq )

    if length > 0:
        gap_data = np.empty(length)
        gap_data[:] = np.NaN
    else:
        gap_data = np.empty(0)
    #fill gap_data with nans
    #print('get_gap_filler.length = ' + str(len(gap_data)))
    #print('get_gap_filler.start_time = ' + str(start_time_microsec))
    #print('get_gap_filler.end_time = ' + str(end_time_microsec))
    return gap_data



class SignalClass:
    '''manages the signal data'''
    def __init__(self, hdf5_data, signal_name):
        self._hdf5_data = hdf5_data
        self._data_reader = MyHdF5signalReaderClass(self._hdf5_data)
        self._data_reader.init_wave_data(signal_name)
        self._signal_name = signal_name
        self._page_start_time = 0
        self._page_len_microsec = 0
        self._page_end_time = 0

    def __init__mock(self, hdf5_data, signal_name):
        #self._hdf5_data = None
        #self._data_reader = None
        self._page_start_time = 0
        self._page_len_microsec = 0
        self._page_end_time = 0


    def get_sampling_freq(self):
        return self._data_reader.get_sampling_freq()

    def get_data_start(self):
        return self._page_start_time

    def get_all_data_start_time(self):
        return self._data_reader.get_all_data_start_time()

    def get_all_data_end_time(self):
        return self._data_reader.get_all_data_end_time()


    def get_data_stream(self, page_start_time=None, page_len_microsec=None):
        '''obtains the full data stream of continuous values, counting only for good quality data'''
        # if arguments are None-s, then the entire stream will be read
        if page_start_time is None:
            page_start_time = self.get_all_data_start_time()
        if page_len_microsec is None:
            page_len_microsec = self.get_all_data_end_time() - page_start_time
        self._page_start_time = page_start_time
        self._page_len_microsec = page_len_microsec
        self._page_end_time = page_start_time + page_len_microsec

        #print('get_data_stream: ._page_start_time=' + str(self._page_start_time) + ', _page_end_time='+ str(self._page_end_time )) # for debugging

        self._data_reader.load_raw_data_set(page_start_time, page_len_microsec)

        # Create empty output array
        complete_data = DataStreamClass()


        #if self._data_reader.is_empty():
        if self._data_reader.is_empty_or_abnormal():
            size = self._calc_page_gap_size()
            complete_data.append_NAN_section(size)

        else:
            # period starts before data : take care of by fillining in with NANs
            size = self._calc_initial_gap_size()
            complete_data.append_NAN_section(size)

            # Process each continuous section, one at a time
            for data_sec in self._data_reader:
                complete_data.append_new_data_section(data_sec)

            # period ends after data: take care of by filling in with NANs
            size = self._calc_end_gap_size()
            complete_data.append_NAN_section(size)

        return complete_data

    def _calc_initial_gap_size(self):

        if self._data_reader.is_normal_quality_point(self._data_reader.get_loaded_data_start_index()):
            first_normal_val_idx = self._data_reader.get_loaded_data_start_index()
        else:
            first_normal_val_idx = self._data_reader.find_next_normal_quality_section_index( self._data_reader.get_loaded_data_start_index())

        if first_normal_val_idx > self._data_reader.get_loaded_data_end_index():
            return 0

        first_normal_val_time =self._data_reader.index_to_time(first_normal_val_idx)

        if self._page_start_time < first_normal_val_time:
            delta_t = first_normal_val_time - self._page_start_time
            float_length = delta_t/ MICROSEC_IN_SEC * self._data_reader.get_sampling_freq()
            return math.floor(float_length)
        return 0

    def _calc_end_gap_size(self):
        loaded_end_index = self._data_reader.get_loaded_data_end_index()

        float_length = 0
        if self._data_reader.is_normal_quality_point(loaded_end_index):
            time_of_last_loaded_val = self._data_reader.index_to_time(loaded_end_index)
            delta_t = self._page_end_time - time_of_last_loaded_val
        else:
            # and when the last data point is NOT normal do this
            current_quality_entry_index = self._data_reader.get_quality_entry_index(loaded_end_index)
            if current_quality_entry_index >= 0:
                current_quality_section_start_time = self._data_reader.get_quality_section_start_time(current_quality_entry_index)

                sample_index, delta_t = self._data_reader.time_to_sample_index(current_quality_section_start_time)
                rel_delta_t = delta_t / MICROSEC_IN_SEC * self._data_reader.get_sampling_freq()
                if rel_delta_t - math.floor(rel_delta_t) > 0:
                    sample_index += 1
                current_quality_section_start_time = self._data_reader.index_to_time(sample_index)

                delta_t = self._page_end_time - current_quality_section_start_time
            else:
                delta_t = self._page_end_time - self._page_start_time
            float_length = 1

        float_length = float_length + delta_t/ MICROSEC_IN_SEC * self._data_reader.get_sampling_freq()
        length = math.floor( float_length )
        if float_length - length == 0:
            length -= 1
        return length

    def _calc_page_gap_size(self):

        if self._data_reader.is_empty_or_abnormal():

            # Find out the index of the last normal value, before the page start
            prev_value_idx, delta_time  = self._data_reader.time_to_sample_index(self._page_start_time)
            if (prev_value_idx >= 0) and self._data_reader.is_normal_quality_point(prev_value_idx):
                last_normal_val_idx_before_page = prev_value_idx
            else:
                qual_entry_index = self._data_reader.get_quality_entry_index( prev_value_idx )
                if qual_entry_index > 0:
                    prev_qual_entry_index= self._data_reader.get_quality_entry_index( qual_entry_index -1)
                    last_normal_val_idx_before_page = self._data_reader.find_end_index_of_current_quality_section(prev_qual_entry_index)
                else:
                    last_normal_val_idx_before_page = -1

            ## takes care of entire page within a gap
            if last_normal_val_idx_before_page >= 0:
                time_of_last_normal_val_before_page = self._data_reader.index_to_time( last_normal_val_idx_before_page )
                page_start_delta_t = self._page_start_time - time_of_last_normal_val_before_page
                page_start_offset_float = page_start_delta_t/MICROSEC_IN_SEC * self._data_reader.get_sampling_freq()
                page_start_offset = math.floor(page_start_offset_float)
                page_end_delta_t = self._page_end_time - time_of_last_normal_val_before_page
                page_end_offset_float = page_end_delta_t/MICROSEC_IN_SEC * self._data_reader.get_sampling_freq()
                page_end_offset = math.floor(page_end_offset_float)
                gap_length = page_end_offset - page_start_offset
                # handle the situation when the page ends after value time (rahter than exactly on the value, which is then excluded)
                if (page_start_offset_float - page_start_offset) == 0:
                    gap_length += 1
                if (page_end_offset_float - page_end_offset) == 0:
                    gap_length -= 1
                return gap_length


            ## takes care of entire page before start of data
            if self._data_reader.is_normal_quality_point(0):
                next_normal_val_idx_after_page = 0
            else:
                next_normal_val_idx_after_page = self._data_reader.find_next_normal_quality_section_index(0)
            if next_normal_val_idx_after_page >= 0:
                time_of_first_normal_val_after_page = self._data_reader.index_to_time( next_normal_val_idx_after_page )
                page_start_delta_t = time_of_first_normal_val_after_page - self._page_start_time
                page_start_offset = math.floor(page_start_delta_t/MICROSEC_IN_SEC * self._data_reader.get_sampling_freq())
                page_end_delta_t = time_of_first_normal_val_after_page - self._page_end_time
                page_end_offset_float =  page_end_delta_t/MICROSEC_IN_SEC * self._data_reader.get_sampling_freq()
                page_end_offset = math.floor(page_end_offset_float)
                gap_length = page_start_offset - page_end_offset
                return gap_length

            # takes care of completely empty (or invalid) data stream
            return math.floor((self._page_end_time - self._page_start_time)/MICROSEC_IN_SEC * self._data_reader.get_sampling_freq())
        return 0


class CacheSignalClass:
    '''Manages tha caching process'''
    def __init__(self, hdf5_data, signal_name, cache_duration_in_hours):
        self._cache_duration_microsec = cache_duration_in_hours * 3600 * 1000 * 1000
        self._signal = SignalClass(hdf5_data, signal_name)
        self._data_cache = None
        self._cache_start_time = 0
        self._sampling_freq  = 0

    def _update_cache(self, start_time):
        self._data_cache = self._signal.get_data_stream(start_time, self._cache_duration_microsec )
        self._cache_start_time = self._signal.get_data_start()
        self._sampling_freq = self._signal.get_sampling_freq()

    def get_all_data_start_time(self):
        return self._signal.get_all_data_start_time()

    def get_all_data_end_time(self):
        return self._signal.get_all_data_end_time()

    def get_data_stream(self, page_start_time, page_len_microsec):

        if (self._data_cache is None)  or (self._cache_start_time > page_start_time) or  (page_start_time + page_len_microsec > self._cache_start_time + self._cache_duration_microsec):
           self._update_cache(page_start_time)

        first_data_index = round((page_start_time - self._cache_start_time) / MICROSEC_IN_SEC * self._sampling_freq)
        length = round(page_len_microsec / MICROSEC_IN_SEC * self._sampling_freq)

        if length > len(self._data_cache.values):
            length = len(self._data_cache.values)

        requested_data = DataStreamClass()
        requested_data.values = self._data_cache.values[first_data_index:first_data_index+length]
        requested_data.sampling_frq = self._sampling_freq
        requested_data.start_time_microsec = page_start_time

        return requested_data






