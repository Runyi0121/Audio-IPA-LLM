for i in English*; do
    newname=$(echo "$i" | sed 's/English/english/')
    mv "$i" "$newname"
done

