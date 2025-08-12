# -*- coding: utf-8 -*-


from datetime import datetime 
import os
from abc import ABCMeta, abstractmethod

import numpy as np

from InfoParser import PatientInfoParser, PatientHadPetMR


class DeltaTimeHandler(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def acquire_scantime(self)->datetime:
        pass
    
    @abstractmethod
    def acquire_radionuclide_starttime(self) -> datetime:
        pass
    
    def calculate_deltatime(self):
        use_constant_time = False
        if not use_constant_time:
            # scantime is the pet scan start time, later
            #scantime = self.acquire_scantime()
            scantime = self.acquire_acquisitiontime()
            # radionuclide_starttime is the drug injected time, earlier
            radionuclide_starttime = self.acquire_radionuclide_starttime() #
            
            # print('acqtime ',self.acquire_acquisitiontime())
            # print('scantime', scantime)
            # print('radio st time', radionuclide_starttime)
            # print('diff time', scantime-radionuclide_starttime)
            # print('deltatime: ',(scantime-radionuclide_starttime).seconds)
            
            # constant_deltatime = datetime.strptime('2016-12-01 00:50:00', "%Y-%m-%d %H:%M:%S") - \
            #                         datetime.strptime('2016-12-01 00:00:00', "%Y-%m-%d %H:%M:%S")
            # print('constant_deltatime: ',
            #       constant_deltatime.seconds)
            
            # print(type(constant_deltatime.seconds))

            return (scantime-radionuclide_starttime).seconds # int
        else:
            print("SUV Calculate use constant time")
            return 3000 # 50 min


class UniteImagingDTHandler(DeltaTimeHandler):
    def __init__(self, time_params) -> None:
        super().__init__()
        self.__time_params = time_params

    def format_datetime(self, datastr:str, time:str)->datetime:
        #print(datastr, time)
        year, month, day = datastr[0:4], datastr[4:6], datastr[6:8]
        hour, min, sec = time[0:2], time[2:4], time[4:6]
        datetime_str = "{year}-{month}-{day} {hour}:{min}:{sec}".format(
            year=year, month=month, day=day, hour=hour, min=min, sec=sec)
        
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

    def acquire_scantime(self)->datetime:
        time_params = self.__time_params
        return self.format_datetime(
            time_params['StudyDate'], 
            time_params['StudyTime']
            )
        
    def acquire_seriestime(self)->datetime:
        time_params = self.__time_params

        return self.format_datetime(
            time_params['StudyDate'], 
            time_params['SeriesTime']
            )
        
    def acquire_acquisitiontime(self)->datetime:
        time_params = self.__time_params
        
        return self.format_datetime(
            time_params['StudyDate'], 
            time_params['AcquisitionTime']
            )

    def acquire_radionuclide_starttime(self)->datetime:
        time_params = self.__time_params
        if time_params['RadiopharmaceuticalStartDateTime'] is None:
            # print(time_params['StudyDate'], time_params['RadiopharmaceuticalStartTime'])
            time_params['RadiopharmaceuticalStartDateTime'] =\
                time_params['StudyDate'] + time_params['RadiopharmaceuticalStartTime']
        return self.format_datetime(
            time_params['RadiopharmaceuticalStartDateTime'],
            time_params['RadiopharmaceuticalStartTime']
             )

class PatientFactorCalculator:
    def __init__(self, patient_params:dict) -> None:
        self.__weight = patient_params['PatientWeight'] # unit:kg
        self.__sex = patient_params['PatientSex'] 
        self.__height = patient_params['PatientSize']*100 \
            if patient_params['PatientSize'] is not None else None # unit:cm
            
        # if patient_params['PatientWeight'] is None:
        #     print("## Set default patient info")
        #     self.set_default()
            
    def set_default(self) -> None:
        if self.__weight is None:
            self.__weight = 60 # unit:kg
        if self.__height is None:
            self.__height = 160 # unit:cm
    
    def __suv_lbm(self)->float:
        weight_coef = {'M':1.1, 'F':1.07}.get(self.__sex)
        div_height_coef = {'M':120, 'F':148}.get(self.__sex)

        # weight unit:kg, weight_height:cm
        lbm = weight_coef*self.__weight-div_height_coef*(
            self.__weight/self.__height)**2 
        
        return lbm*1000  # unit:g

    def __suv_bw(self)->float:
        return self.__weight*1000 # unit:g

    def __suv_bsa(self)->float:
        return self.__weight**0.425*self.__height**0.725*71.84

    def acquire_patient_factor(self, suv_type:str)->float:
        patient_factor = {
            'suvlbm':self.__suv_lbm,
            'suvbw':self.__suv_bw
        }.get(suv_type)

        return patient_factor()


class DelaTimeCalculator:
    def __init__(self) -> None:
        pass

    def calculate_deltatime(self, time_handler:DeltaTimeHandler):
        return time_handler.calculate_deltatime()


class SUVCalculator(object):
    def __init__(self, info_parser:PatientInfoParser, manufacturer:str='UIH', suv_info=None) -> None:
        if info_parser:
            self.calculation_params= info_parser.acquire_suv_calculation_params()
        else:
            self.calculation_params = suv_info
        
        self.__convert_radionuclide_unit()
        self.__DTHander = {
            'UIH':UniteImagingDTHandler(self.calculation_params),
            }.get(manufacturer)
        
        if self.calculation_params['PatientSize'] is None:
            self.suv_type = 'suvbw'
        else:
            self.suv_type = None

    def __convert_radionuclide_unit(self)->None:
        radionuclide_total_dose = self.calculation_params['RadionuclideTotalDose']
        #TODO: It can check if the unit is BQML
        if radionuclide_total_dose<10000:
            # convert unit from MBq to Bq
            self.calculation_params[
                'RadionuclideTotalDose'] = radionuclide_total_dose*1000000
        else:
            # The raw unit is Bq
            pass

    def calculate_injected_dose(self):
        radionuclide_total_dose = self.calculation_params['RadionuclideTotalDose']
        radionuclide_half_life = self.calculation_params['RadionuclideHalfLife']
        calculator = DelaTimeCalculator()
        delta_time = calculator.calculate_deltatime(self.__DTHander) # unit:sec
        # print(delta_time)
        radionuclide_total_dose_factor = np.exp(-0.693*delta_time/radionuclide_half_life)

        # print('r_total_dose', radionuclide_total_dose)
        return radionuclide_total_dose_factor*radionuclide_total_dose
        
    def calculate_avgactivity(self, voxel_intensity_in_roi:np.ndarray):
        img_scale_factor = float(self.calculation_params['RescaleSlope'])
        img_intercept_factor = float(self.calculation_params['RescaleIntercept'])

        img_scale_factor = max(1, img_scale_factor)
        avgactivity = voxel_intensity_in_roi*img_scale_factor+img_intercept_factor
        
        return avgactivity

    def calculate_patient_factor(self, suv_type:str='suvlbm'):
        assert suv_type in ['suvlbm', 'suvbw', 'suvbsa'], 'Unknow suv type'
        # if the dicom is not save patient size
        if self.suv_type is None:
            self.suv_type = suv_type
        
        factor_calculator = PatientFactorCalculator(self.calculation_params)
        return factor_calculator.acquire_patient_factor(self.suv_type) # unit: g

    def calculate_suv(self, voxel_intensity_in_roi:np.ndarray, suv_type:str='suvlbm'):
        assert suv_type in ['suvlbm', 'suvbw', 'suvbsa'], 'Unknow suv type'
        
        # if the dicom is not save patient size
        if self.suv_type is None:
            self.suv_type = suv_type
        # print(self.suv_type)
        avg_activity = self.calculate_avgactivity(voxel_intensity_in_roi)
        injected_dose = self.calculate_injected_dose()
        patient_factor = self.calculate_patient_factor(self.suv_type) # unit: g
        
        # print('pf', patient_factor)
        # print('id', injected_dose)
        # print(patient_factor/injected_dose)
        
        return (avg_activity/injected_dose)*patient_factor
        

if __name__ == "__main__":
    print('----TestUnit----')
    # Get absolute path of the patient dir
    test_patient_dir = os.path.realpath("../SampleData/patient_petmr")
    if os.path.exists(test_patient_dir):
        PatientParser = PatientHadPetMR(test_patient_dir)
    else:
        print('Patient folder is empty!')
    print(test_patient_dir)
    assert PatientParser, 'The patient parser init failed'

    calculation_params = PatientParser.acquire_suv_calculation_params()
    calculator = DelaTimeCalculator()
    DTHander = {
            'UIH':UniteImagingDTHandler(calculation_params),
            }.get('UIH')
    
    delatime = calculator.calculate_deltatime(
        UniteImagingDTHandler(calculation_params)
        )
    
    # result should be 40.633(min) or 2438(sec)
    print(delatime)

    patient_suv_calculator = SUVCalculator(PatientParser)

    #result should be 50363730.79305495
    print(patient_suv_calculator.calculate_injected_dose())

    #result should be 56653.97923875434
    print(patient_suv_calculator.calculate_patient_factor())

    #input = [25,103,5645] output = [0.0493, 0.2030, 11.1265]
    test_input_intensity = np.array([25,103,5645])
    print(patient_suv_calculator.calculate_suv(test_input_intensity))