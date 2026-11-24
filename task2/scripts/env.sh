# source this first. GO_ROOT holds src/ data/ results/ logs/ sub/ katago/ ; copy go/src/*.py into $GO_ROOT/src
export GO_ROOT=${GO_ROOT:-$HOME/go_run}
export PY=${PY:-python3}
export PYTHONHASHSEED=0
export GPU=${GPU:-0}                       # CUDA device index used by all training/inference below
mkdir -p $GO_ROOT/{src,data,results,logs,sub,katago}
cd $GO_ROOT
