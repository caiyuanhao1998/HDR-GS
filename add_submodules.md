How to add submodules in github:
```sh
# Firstly, remove the original .gitmodules file
rm .gitmoudles

# Then create a new .gitmodules file
touch .gitmodules

# add submodules one-by-one as
git submodule add https://gitlab.inria.fr/sibr/sibr_core.git SIBR_viewers

git submodule add https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization

git submodule add https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
```