# Prepare test file s2.txt
sed -f sub.sed s2.txt |tr [:upper:] [:lower:] | tr -s ' ' > s2.nor
tr ' ' '\012' <s2.nor >s2.wrd
sort s2.wrd | uniq -c > s2.1
awk '{if(NR>1) print ant " " $1; ant=$1}' s2.wrd | sort | uniq -c | grep -v '</s> <s>' | grep -v '<s> </s>' > s2.2
awk '{print $2 "_" $3}' s2.2 | sort > s2.big

# Prepare test file s3.txt
sed -f sub.sed s3.txt |tr [:upper:] [:lower:] | tr -s ' ' > s3.nor
tr ' ' '\012' <s3.nor >s3.wrd
sort s3.wrd | uniq -c > s3.1
awk '{if(NR>1) print ant " " $1; ant=$1}' s3.wrd | sort | uniq -c | grep -v '</s> <s>' | grep -v '<s> </s>' > s3.2
awk '{print $2 "_" $3}' s3.2 | sort > s3.big

# compute number of occurrences of bigrams for the training set
join -1 2 -2 2 -o 2.2,2.3,1.1,2.1 s1.1 s1.2 > s1.bn

# compute bigram probabilities for the training set
awk '{print $1 "_" $2 " " ($4+1)/($3+1)} " " $3' s1.bn | sort > s1.big

join -1 1 -2 1 -a 2 s1.big s2.big | awk 'BEGIN{prod=1}{if(NF==2) prod=prod*$2; else prod=1/($3+1)}END{print prod}' > prob_smooth_s2.txt
join -1 1 -2 1 -a 2 s1.big s3.big | awk 'BEGIN{prod=1}{if(NF==2) prod=prod*$2; else prod=1/($3+1)}END{print prod}' > prob_smooth_s3.txt

rm -r s2.nor
rm -r s2.wrd
rm -r s2.1
rm -r s2.2
rm -r s2.big
rm -r s1.bn
rm -r s1.big
rm -r s3.nor
rm -r s3.wrd
rm -r s3.1
rm -r s3.2
rm -r s3.big