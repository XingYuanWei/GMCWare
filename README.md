# GMCWare
GMCWare: A Greedy Modularity Community-based Simplification Algorithm for Malware Detection
**First** You need to download bert-uncased pytorch_model.bin file from hugging face into bert folder.

## How to run this.

1. first conda envriment<br>
   cd GMCWare<br>
   pip install -r requirements.txt <br>
3. if (you want to train a new model）:<br>
       You need to prepare a pair of benign and malware.<br>
       Then you get the Dataset from 'Modularity_Communities' folder script,<br>
       and get Pytorch Dataset from 'GMCDataset.py' script.<br>
       use Trainer.py to train your Model.<br>
   else only test model test:<br>
       cd GNNModel Folder<br>
       Then change Trainer.py, you can understand the code easily by the comments <br>
       and load the model from 'Model_GraphDoatGAt.pth' or 'Model_GraphGAT.pth', 'Model_GraphSAGE.pth'... <br>
       run Trainer.py<br>
       GMCWare_Test_Data includes a Dataset we open-source in GMC_data folder, only 50 per class <br>
