export LC_ALL=C
export LANG=pt_PT

# normalize file
sed -f sub.sed s1b.txt |tr [:upper:] [:lower:] | tr -s ' ' > s1.nor
# get one word per line
tr ' ' '\012' <s1.nor >s1.wrd

# build unigram file
sort s1.wrd | uniq  -c >s1.1

# build bigram file
awk '{if(NR>1) print ant " " $1; ant=$1}' s1.wrd | sort | uniq -c | grep -v '</s> <s>' | grep -v '<s> </s>' > s1.2

# join the two files
#cat s1.1 s1.2 > ngrams_s1prev.txt
sort -r s1.1 > s1.1.txt
sed -i '11,$ d' s1.1.txt
sort -r s1.2 > s1.2.txt
sed -i '11,$ d' s1.2.txt

echo '10 most common unigrams' >> aux1.txt
echo '10 most common bigrams' >> aux2.txt
cat aux1.txt s1.1.txt aux2.txt s1.2.txt > ngrams_s1.txt

rm -r aux1.txt
rm -r aux2.txt
rm -r s1.1.txt
rm -r s1.2.txt
rm -r s1.nor
rm -r s1.wrd