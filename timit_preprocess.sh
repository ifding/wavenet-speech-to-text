if [ "$#" -ne 1 ]; then
    echo "Usage : ./timit_preprocess.sh <timit folder>"
fi
echo 'Transfering raw TIMIT wave file format from NIST to RIFF.'
echo ' '
find $1 -name '*.wav' | parallel -P20 sox {} '{.}.WAV'
# MFCC
python3 utils/preprocess.py $1 timit_mfcc_39