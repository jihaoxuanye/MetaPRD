### Introduction
For pre-trained GCN.

### Usage

Please firstly generate the metadata for GCN pre-training in follows:
cd ../..
python gen_meta_gcn_data.py -d <dataset_name> --data-dir <data_path>, e.g. python gen_meta_gcn_data.py -d market1501 --data-dir ../../data

Second, we obtain the pre-training model for GCN in follows:
cd ../..
python gcn_pre_train.py -d <dataset_name> --data-dir <data_path>, e.g. python gen_meta_gcn_data.py -d market1501 --data-dir ../../data

we also provide a example on Market-1501:
metadata (market_meta_data.pt) from [Baidu](https://pan.baidu.com/s/1gwrGsIWL_gZEEi0prK_5UA), code: 0wev; 
                               or from [Google](https://drive.google.com/file/d/1GqUpVGY8NPwbNW9JNT2YfKG6briKqd7X/view?usp=sharing)
pre-trained (pretrained_meta_market_gcn.pth.tar) from [Baidu](https://pan.baidu.com/s/18zxexpYoz8H7cyvkLtxVNQ), code: h2up; 
                                                 or from [Google Drive](https://drive.google.com/file/d/18_cCe4kVV_zGObsPbjcXkt5JbjMBQcGo/view?usp=sharing)
