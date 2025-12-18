import pydicom
import matplotlib as plt

class DICOMReader:

    def __init__(self, dicom_path:str):
        self.dicom_path = dicom_path
        self.dataset = None

    def  load(self):
            self.dataset=pydicom.dcmread(self.dicom_path)
            return self.dataset
    def show_metadata(self):
              if self.dataset is None:
                raise ValueError("DICOM file not loaded")

              print("Patient ID:", self.dataset.get("PatientID", "N/A"))
              print("Modality:", self.dataset.get("Modality", "N/A"))
              print("Study Date:", self.dataset.get("StudyDate", "N/A"))    

    def show_image(self):
        if self.dataset is None:
            raise ValueError("DICOM file not loaded")

        plt.imshow(self.dataset.pixel_array, cmap="gray")
        plt.title("DICOM Image")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    path = "sample.dcm"  
    reader = DICOMReader(path)
    reader.load()
    reader.show_metadata()
    reader.show_image()   