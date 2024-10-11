**DeepPTCL** \
To simulate a case of new patient tumor, we additionally tested our model in the most challenging  leave cell out to validate cell line specific performance when the cell line was not part of the training set. \
\
In this way, the cell lines in the test dataset are the most different from those in the training dataset. The leave cell out is a robust method to test the generalization of the our model.\
\
We compared to the state-of-art model based on the metric of Aera Under ROC (AUROC) and Precision Under ROC (PRAUC). Our model DeepPTCL has obtained best performance (AUROC= 0.80, PRAUC =0.54) compared to DeepSynergy(AUROC = 0.72 , PRAUC = 0.36). \
\
To see how well our model predicted the synergy ranking if different combinations of drugs within cell lines, we also calculated he Spearman correlation between DeepPTCL predictions and the actual synergy scores by cell lines. We observed the ranking of drug combinations were fairly consistent across all cell lines, predominantly between 0.6 and 0.75.

![Blank diagram - Page 1 (2)](https://github.com/user-attachments/assets/5d416b19-6c9a-4306-99b1-1fd1ad6f87cd)
