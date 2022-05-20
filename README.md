# GE_pj07_sdgs_nlp

This is a code for understanding SDGs (Sustainable Development Goals) using a Natural Language Processor.
The model is for estimating Global Goals, vectorizing sematics and visualizing interklinkages between SDGs related to the input texts.

The original article is the below.
'T. Matsui, K. Suzuki, K. Ando, Y. Kitai, C. Haga, N. Masuhara, S. Kawakubo: A Natural Language Processing Model for Supporting Sustainable Development Goals: Translating Semantics, Visualizing Nexus, and Connecting Stakeholders, Sustainability Science. 2021.12. <a href="https://link.springer.com/article/10.1007/s11625-022-01093-3">DOI:10.1007/s11625-022-01093-3'</a>.

Aknowledgement and Usuful reference for Japanese:
1. Natural Languate Processing by BERT, https://www.ohmsha.co.jp/book/9784274227264/
2. Natural language processing by Transformer, https://www.asakura.co.jp/detail.php?book_code=12265
3. Natural language processing by AI, https://www.c-r.com/book/detail/1435
4. Japanese pretrained BERT, https://github.com/cl-tohoku/bert-japanese

# UPDATE
## 18.05.2022
You can run the pre-trained model weight in ENG version with 'sdgs_translator_for_git.py'. You can download the model weight from <a href = "https://www.dropbox.com/s/owzc1u6khhpe6js/best_model_gpu.pth?dl=0">here</a>. Proofed on pytorch==1.8.2, transformers==4.16.1. The accuracy of this version is still low, this is due to the low-divergence of the training dataset. Please enjoy the model just for boosting your imagination.

-----------------------------------------------------------------
  precision    recall  f1-score   support
-----------------------------------------------------------------
           0       0.56      0.25      0.34        40
           1       0.52      0.32      0.39        47
           2       0.68      0.45      0.54        42
           3       0.55      0.16      0.25        37
           4       0.70      0.62      0.66        50
           5       0.69      0.53      0.60        66
           6       0.73      0.51      0.60        65
           7       0.64      0.16      0.26        43
           8       0.45      0.28      0.35        46
           9       0.67      0.26      0.38        53
          10       0.62      0.28      0.38        36
          11       0.67      0.33      0.44        60
          12       0.76      0.51      0.61        73
          13       0.72      0.58      0.64        45
          14       0.55      0.41      0.47        44
          15       0.65      0.40      0.49        50
          16       0.75      0.48      0.59        50
-----------------------------------------------------------------
-----------------------------------------------------------------
   micro avg       0.66      0.40      0.50       847
   macro avg       0.64      0.38      0.47       847
weighted avg       0.65      0.40      0.49       847
 samples avg       0.40      0.40      0.40       847
 -----------------------------------------------------------------
