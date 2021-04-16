pip3 -q install pydicom
pip -q uninstall -y kaggle
pip -q install --upgrade pip
pip3 -q install --upgrade kaggle

mkdir /root/.kaggle
echo '{"username":"hector6298","key":"724778e3045b27ede8002c9f01b9da72"}' > /root/.kaggle/kaggle.json

git  clone https://github.com/ieee8023/covid-chestxray-dataset.git
git  clone https://github.com/agchung/Figure1-COVID-chestxray-dataset
git  clone https://github.com/agchung/Actualmed-COVID-chestxray-dataset
kaggle datasets download -d "tawsifurrahman/covid19-radiography-database"
kaggle competitions download -c "rsna-pneumonia-detection-challenge" 
unzip rsna-pneumonia-detection-challenge.zip
rm rsna-pneumonia-detection-challenge.zip
unzip covid19-radiography-database.zip
rm covid19-radiography-database.zip
mkdir data
mkdir data/train
mkdir data/test
mkdir /content/logs

python3 "/content/drive/My Drive/COVID-Net-master/COVID-Net-master/create_covidx_v3.py"