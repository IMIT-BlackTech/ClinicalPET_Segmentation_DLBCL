# -*- coding: utf-8 -*-



from abc import ABCMeta, abstractmethod

import pydicom
from pydicom.dataset import Dataset

class PyDicomDatasetParser(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def parser_attribute(dicom_data:Dataset, required_attribute:list)->str:
        """Return the dicom :dict:'{attribute: val}'"""
        # acquire vals by the Value Representations(VR)
        if isinstance(required_attribute[0], str):
            attr_val_dict = [(vr_name, \
                PyDicomDatasetParser.parser_by_vr(dicom_data ,vr_name))\
                 for vr_name in required_attribute]
        # acquire vals by the (Group Number, Element Number)
        elif isinstance(required_attribute[0], tuple):
            attr_val_dict = [(tag_key, \
                PyDicomDatasetParser.parser_by_tag(dicom_data, tag_key))\
                 for tag_key in required_attribute]
        return dict(attr_val_dict)

    @staticmethod
    def parser_by_vr(dicom_data:Dataset, vr_name:str)->str:
        """Return the dicom value by attribute Tag"""

        try:
            attr_value = getattr(dicom_data, vr_name)
        except AttributeError as e:
            attr_value = None
        return attr_value

    @staticmethod
    def parser_by_tag(dicom_data:Dataset, tag_key:tuple)->str:
        """Return the dicom value by tag (Group Number, Element Number) """

        assert [isinstance(key_tag, str) for key_tag\
                 in tag_key]==[True, True],'Need string as key tag'
        # TagConvert 16-base str --> 10-base int --> 16-base int
        tag_id = tuple(map(hex,
                    map(int, tag_key, [16, 16])
                    )
                )
        try:
            attr_value = dicom_data[tag_id].value
        except KeyError as e:
            attr_value = None
        return attr_value


class MedicalImageParser(metaclass=ABCMeta):
    """An abstract class for Medical Image"""

    def __init__(self) -> None:
        """Create a new :class: 'MedicalImageParser' instance."""

        super().__init__()

    def setup(self, single_dicom_image_path:str):
        return self.input_dicom_file(single_dicom_image_path)

    @abstractmethod
    def input_dicom_file(self, single_dicom_image_path:str) -> None:
        self.__dicom_data = pydicom.dcmread(single_dicom_image_path)
        # self.__accessible_vrs = self.__dicom_data.dir()
        # self.__accessible_tags = list(self.__dicom_data.keys())
        return self

    @abstractmethod
    def parser_attribute(self, required_attribute:list)->dict:
        """Return the dicom :dict:'{attribute: val}'"""

        return PyDicomDatasetParser.parser_attribute(self.__dicom_data,\
             required_attribute)



class PetImageParser(MedicalImageParser):
    img_type = 'pet'
    def __init__(self) -> None:
        super().__init__()
        self.__radionuclide_data = []

    def input_dicom_file(self, single_dicom_image_path: str) -> None:
        return super().input_dicom_file(single_dicom_image_path)

    def parser_attribute(self, required_attribute: list) -> dict:
        return super().parser_attribute(required_attribute)

    def parser_radionuclide(self, required_radionuclide_attrs:list) -> dict:
        """Return the required radionuclide attributes"""

        if not self.__radionuclide_data:
            self.__radionuclide_data = self.parser_attribute(
                ['RadiopharmaceuticalInformationSequence']
            )['RadiopharmaceuticalInformationSequence'][0]
        return PyDicomDatasetParser.parser_attribute(self.__radionuclide_data,\
             required_radionuclide_attrs)


class MRIImageParser(MedicalImageParser):
    img_type = 'mri'
    def __init__(self) -> None:
        super().__init__()

    def input_dicom_file(self, single_dicom_image_path: str) -> None:
        return super().input_dicom_file(single_dicom_image_path)

    def parser_attribute(self, required_attribute: list) -> dict:
        return super().parser_attribute(required_attribute)


class CTImageParser(MedicalImageParser):
    def __init__(self) -> None:
        super().__init__()

    def input_dicom_file(self, single_dicom_image_path: str) -> None:
        return super().input_dicom_file(single_dicom_image_path)

    def parser_attribute(self, required_attribute: list) -> dict:
        return super().parser_attribute(required_attribute)


if __name__ == "__main__":
    print('----TestUnit----')
    import os, glob
    test_subpath_list = [
        "../SampleData/patient_petmr/pet", # pet dicom files
        "../SampleData/patient_petmr/mr", # mr dicom files
        "../SampleData/patient_petmr/all_data.mat" # all_data.mat in matlab
        ]
    # Get absolute path of the test dicom subfoler 
    test_pet_dicom_folder, test_mr_dicom_folder, test_mat_data_path =\
         list(map(os.path.realpath, test_subpath_list))
    # Check foleder and get the pet dicom files
    if os.path.exists(test_pet_dicom_folder):
        test_pet_dicom_list = glob.glob(
            os.path.join(test_pet_dicom_folder, '*.dcm')
        )
    assert test_pet_dicom_list, 'The sample folder is empty!'
    # Create the pet parser instance
    pet_parser_instance = PetImageParser()
    print(
        pet_parser_instance.input_dicom_file(test_pet_dicom_list[0]
            ).parser_attribute(
            ['PatientName', 'PatientSex', 'PatientWeight', 
            'PatientID', 'Units', 'Manufacturer']
            )
    )
    
    suvlbm_calculation_attrs = [
        'RadiopharmaceuticalStartDateTime', 'RadionuclideHalfLife',
        'RadiopharmaceuticalStartTime', 'Radiopharmaceutical', 
        'RadionuclideTotalDose'
    ]
    print(pet_parser_instance.parser_radionuclide(suvlbm_calculation_attrs))