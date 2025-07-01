{ pkgs ? import <nixpkgs> {
		config.allowUnfree = true;
		} }:

pkgs.mkShellNoCC {
  packages = with pkgs; [
    python3
    python3Packages.pip
    uv
  ];

  shellHook = ''
	export UV_CACHE_DIR="/home/jahan/ssd3/uvcache/"
	export UV_PROJECT_ENVIRONMENT="/home/jahan/ssd3/venvs/icu-los"
	export HF_HOME="/home/jahan/ssd3/hfcache/"
	alias venv="source /home/jahan/ssd3/venvs/icu-los/bin/activate"
	if [ -f ./.env.public ]; then
    export $(grep -v '^#' ./.env.public | xargs)
  fi
	if [ -f ./.env.private ]; then
    export $(grep -v '^#' ./.env.private | xargs)
  fi
  '';
}

