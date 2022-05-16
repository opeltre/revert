tags=("Baseline" "Plateau" "Infusion")

if [ -d $INFUSION_DATASETS/no_shunt ]; then
    echo "no_shunt dataset directory exists"
else
    python scripts-infusion/filter_noshunt.py
fi

function extractPulses () {
    python scripts-infusion/extract_timestamps.py $1
    for tag in ${tags[*]}; do
       python scripts-infusion/extract_pulses.py $1 $tag 
    done
}

extractPulses $1
