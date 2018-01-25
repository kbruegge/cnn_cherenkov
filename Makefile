gamma_phs=./data/photonstream/gamma/DONE
proton_phs=./data/photonstream/proton/DONE
crab_phs=./data/photonstream/crab/DONE


gamma_images=./data/gamma_images.hdf5
proton_images=./data/proton_images.hdf5
crab_images=./data/crab_images.hdf5

all: $(gamma_images) $(proton_images) $(crab_images)

$(crab_images):$(crab_phs)
	python convert.py --yes -n 24 ./data/photonstream/crab/*phs.jsonl.gz $(crab_images)

$(gamma_images):$(gamma_phs)
	python convert.py --yes -n 24 ./data/photonstream/gamma/*phs.jsonl.gz $(gamma_images)

$(proton_images):$(proton_phs)
	python convert.py --yes -n 24 ./data/photonstream/proton/*phs.jsonl.gz $(proton_images)



$(gamma_phs):
	wget -nH --cut-dirs=5 -e robots=off -r -l 1 --no-parent -P ./data/photonstream/gamma -A '*ch.gz' https://ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/
	wget -nH --cut-dirs=5 -e robots=off -r -l 1 --no-parent -P ./data/photonstream/gamma -A '*.phs.jsonl.gz' https://ihp-pc41.ethz.ch/public/phs/sim/gamma/gustav/
	touch $(gamma_phs)

$(proton_phs):
	wget -nH --cut-dirs=5 -e robots=off -r -l 1 --no-parent -P ./data/photonstream/proton -A '*ch.gz' https://ihp-pc41.ethz.ch/public/phs/sim/proton/uwe/
	wget -nH --cut-dirs=5 -e robots=off -r -l 1 --no-parent -P ./data/photonstream/proton -A '*.phs.jsonl.gz' https://ihp-pc41.ethz.ch/public/phs/sim/proton/uwe/
	touch $(proton_phs)

$(crab_phs):
	wget -nH --cut-dirs=3 -e robots=off -r -l 1 --no-parent -P ./data/photonstream/crab -A '*.phs.jsonl.gz' https://ihp-pc41.ethz.ch/public/phs/public/
	touch $(crab_phs)
