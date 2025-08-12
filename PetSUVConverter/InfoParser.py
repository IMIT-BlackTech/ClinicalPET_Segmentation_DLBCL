# -*- coding: utf-8 -*-



import os
import glob
from abc import ABCMeta, abstractmethod

from MedicalImageParser import  PetImageParser

class PatientInfoParser(metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self)->str:
        return self.__patient_name

    @name.setter
    def name(self, value:str)->str:
        self.__patient_name = value

    @abstractmethod
    def mount_descriptive_dicom(self)->None:
        pass
    
    
class PatientInfoReader(PatientInfoParser):
    def __init__(self, patient_dir:str) -> None:
        super().__init__()
        self._dicom_files = patient_dir  
        self.mount_descriptive_dicom()
        
    def mount_descriptive_dicom(self)->None:   
        pet_descriptive_dicom_files = glob.glob(
            os.path.join(self._dicom_files, '*.dcm')
        )
        assert pet_descriptive_dicom_files, \
            'The sample folder {:s} is empty!'.format(self._dicom_files)
        
        # print(pet_descriptive_dicom_files[0])
        # print(pet_descriptive_dicom_files[-1])
        self.PetImageParser = PetImageParser()
        self.PetImageParser.input_dicom_file(pet_descriptive_dicom_files[-1])
        self.name = self.PetImageParser.parser_attribute(['PatientName'])
        
    def read(self, dicom_tags:list)-> dict:
        dicom_values = self.PetImageParser.parser_attribute(dicom_tags)
        
        return dicom_values


class PatientHadPetMR(PatientInfoParser):
    def __init__(self, patient_dir:str) -> None:
        super().__init__()
        self._dicom_files = patient_dir
        
        # self._dicom_files = {
        #     'pet': os.path.join(patient_dir, 'pet'),
        #     'mr': os.path.join(patient_dir, 'mr')
        # }
        
        # assert os.path.exists(self._dicom_files['pet']), 'Patient Pet Image folder missing!'
        # assert os.path.exists(self._dicom_files['mr']), 'Patient MR Image folder missing!'

        self.mount_descriptive_dicom()

    def mount_descriptive_dicom(self)->None:
        # pet_descriptive_dicom_files = glob.glob(
        #     os.path.join(self._dicom_files['pet'], '*.dcm')
        # )
        # assert pet_descriptive_dicom_files, \
        #     'The sample folder {:s} is empty!'.format(self._dicom_files['pet'])
        pet_descriptive_dicom_files = glob.glob(
            os.path.join(self._dicom_files, '*.dcm')
        )
        assert pet_descriptive_dicom_files, \
            'The sample folder {:s} is empty!'.format(self._dicom_files)
        
        # print(pet_descriptive_dicom_files[0])
        # print(pet_descriptive_dicom_files[-1])
        self.PetImageParser = PetImageParser()
        self.PetImageParser.input_dicom_file(pet_descriptive_dicom_files[-1])
        self.name = self.PetImageParser.parser_attribute(['PatientName'])

    def acquire_suv_calculation_params(self)->dict:
        '''Return pet for suvlbm calculation'''
        assert self.name, 'Need patient init process'
        radionuclide_attrs = [
            'RadiopharmaceuticalStartDateTime', 'RadionuclideHalfLife',
            'RadiopharmaceuticalStartTime', 'Radiopharmaceutical', 
            'RadionuclideTotalDose'
        ]
        scan_attrs = ['StudyTime', 'StudyDate', 'SeriesTime', 'AcquisitionTime']
        img_attrs = ['RescaleSlope', 'RescaleIntercept']
        patient_attrs = ['PatientSex', 'PatientWeight', 'PatientSize', 'Units']

        suv_caculation_attrs = [*scan_attrs, *img_attrs, *patient_attrs]
        
        suv_caculation_params = {
            **self.PetImageParser.parser_attribute(suv_caculation_attrs),
            **self.PetImageParser.parser_radionuclide(radionuclide_attrs)
            }

        return suv_caculation_params


if  __name__ == "__main__":
    print('----TestUnit----')
    import os, glob, datetime
    # Get absolute path of the patient dir
    test_patient_dir = os.path.realpath("../SampleData/patient_28f_petmr/pet")
    if os.path.exists(test_patient_dir):
        PatientParserForSUVlbm = PatientHadPetMR(test_patient_dir)
    else:
        print('Patient folder is empty!')
    
    assert PatientParserForSUVlbm, 'The patient parser init failed'

    calculation_params = PatientParserForSUVlbm.acquire_suv_calculation_params()

    print(calculation_params)
    
    patient_attrs = ['PatientName','PatientSex', 'PatientWeight', 'PatientSize', 
                     'PatientID', 'PatientAge','StudyDate', 'StudyTime']
    
    radionuclide_attrs = ['Radiopharmaceutical']
    
    patient_info = PatientParserForSUVlbm.PetImageParser.parser_attribute(patient_attrs)
    patient_info = {**patient_info,**PatientParserForSUVlbm.PetImageParser.parser_radionuclide(
        radionuclide_attrs)}
    
    scantime = patient_info['StudyDate'][6:8]
    year = patient_info['StudyDate'][0:4]
    month = patient_info['StudyDate'][4:6]
    day = patient_info['StudyDate'][6:8]

    hour = patient_info['StudyTime'][0:2]
    mins = patient_info['StudyTime'][2:4]
    sec = patient_info['StudyTime'][4:6]
    
    timestr = "{year}-{month}-{day} {hour}:{mins}:{sec}".format(
                year=year, month=month, day=day, hour=hour, mins=mins, sec=sec)
    
    patient_info['examTime'] = timestr
    patient_attrs.remove('StudyDate')
    patient_attrs.remove('StudyTime')
    patient_attrs.append('examTime')
    patient_attrs.extend(radionuclide_attrs)
    
    print(patient_info)

    scantime = calculation_params['StudyDate'][6:8]
    year = calculation_params['StudyDate'][0:4]
    month = calculation_params['StudyDate'][4:6]
    day = calculation_params['StudyDate'][6:8]

    hour = calculation_params['StudyTime'][0:2]
    mins = calculation_params['StudyTime'][2:4]
    sec = calculation_params['StudyTime'][4:6]

    timestr = "{year}-{month}-{day} {hour}:{mins}:{sec}".format(
        year=year, month=month, day=day, hour=hour, mins=mins, sec=sec)
    print(datetime.datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S"))