tags=("Baseline" "Plateau" "Infusion")

function filterShunt () {
    if [ -d $INFUSION_DATASETS/no_shunt ]; then
        echo "no_shunt dataset directory exists"
    else
        python scripts-infusion/filter_noshunt.py
    fi
}

function extractPulses () {
    dirname=${1:-no_shunt}
    python scripts-infusion/extract_timestamps.py $dirname
    python scripts-infusion/extract_results.py $dirname
    for tag in ${tags[*]}; do
       python scripts-infusion/extract_pulses.py $dirname $tag 
    done
}

filterShunt
extractPulses $1
