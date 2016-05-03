for file in `ls \`dirname "${BASH_SOURCE[0]}"\`/run_*.sh`
#for file in `ls \`dirname "${BASH_SOURCE[0]}"\`/*NEW*`
do
  echo "$file"
  #cp "$file" "$file"_NEWBACKEND
  . "$file"
  #sed -i -e 's/gpu/cuda/g' "$file"
done
