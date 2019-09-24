# deepscan-brats
This repository hosts the contributor source files for the deepscan-brats model. ModelHub integrates these files into an engine and controlled runtime environment. A unified API allows for out-of-the-box reproducible implementations of published models. For more information, please visit [www.modelhub.ai](http://modelhub.ai/) or contact us [info@modelhub.ai](mailto:info@modelhub.ai).

### Info 
This model needs a GPU to run. Please follow the quickstart instructions for GPU on our website to find out how to set up your system: [Quickstart Docs](https://modelhub.readthedocs.io/en/latest/quickstart.html)

## meta
| | |
|-|-|
| id | 20fb032f-7a5b-4a2b-9ab8-367ee53030ea | 
| application_area | Medical Imaging, Segmentation | 
| task | Brain Tumor Segmentation | 
| task_extended | Brain tumor segmentation for the BraTS 18 challenge | 
| data_type | Nifti-1 volumes | 
| data_source | www.braintumorsegmentation.org | 
## publication
| | |
|-|-|
| title | Ensembles of Densely-Connected CNNs with Label-Uncertainty for Brain Tumor Segmentation | 
| source | International MICCAI Brainlesion Workshop | 
| url | https://link.springer.com/chapter/10.1007/978-3-030-11726-9_40 | 
| year | 2018 | 
| authors | Richard McKinley, Raphael Meier, Roland Wiest | 
| abstract | We introduce a new family of classifiers based on our previous DeepSCAN architecture, in which densely connected blocks of dilated convolutions are embedded in a shallow U-net-style structure of down/upsampling and skip connections. These networks are trained using a newly designed loss function which models label noise and uncertainty. We present results on the testing dataset of the Multimodal Brain Tumor Segmentation Challenge 2018. | 
| google_scholar | https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&as_ylo=2018&q=Ensembles+of+densely-connected+CNNs+with+label-uncertainty+for+brain+tumor+segmentation&btnG= | 
| bibtex | @inproceedings{mckinley2018ensembles,title={Ensembles of densely-connected CNNs with label-uncertainty for brain tumor segmentation},author={McKinley, Richard and Meier, Raphael and Wiest, Roland},booktitle={International MICCAI Brainlesion Workshop},pages={456--465},year={2018},organization={Springer}} | 
## model
| | |
|-|-|
| description | Densely-Connected CNNs | 
| provenance |  | 
| architecture | CNN | 
| learning_type | Supervised | 
| format | .pth.tar | 
| I/O | model I/O can be viewed [here](contrib_src/model/config.json) | 
| license | model license can be viewed [here](contrib_src/license/model) | 
## run
To run this model and view others in the collection, view the instructions on [ModelHub](http://app.modelhub.ai/).
## contribute
To contribute models, visit the [ModelHub docs](https://modelhub.readthedocs.io/en/latest/).
