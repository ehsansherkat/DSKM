# Fast and Simple Deterministic Seeding of KMeans for Text Document Clustering

KMeans is one of the most popular document clustering algorithms. It is usually initialized by random seeds that can drastically impact the final algorithm performance. There exists many random or order-sensitive methods that try to properly initialize KMeans but their problem is that their result is non-deterministic and unrepeatable. Thus KMeans needs to be initialized several times to get a better result, which is a time-consuming operation. In this paper, we introduce a novel deterministic seeding method for KMeans that is specifically designed for text document clustering. Due to its simplicity, it is fast and can be scaled to large datasets. Experimental results on several real-world datasets demonstrate that the proposed method has overall better performance compared to several deterministic, random, or order-sensitive methods in terms of clustering quality and runtime.

Read more: https://link.springer.com/chapter/10.1007/978-3-319-98932-7_7

Run test.py for a sample input and output.

Please cite to the following paper if you used the code:
```
@misc{CLEF_Sherkat_2018,
   author="Sherkat, Ehsan
          and Velcin, Julien
          and Milios, Evangelos E.",
   title="Fast and Simple Deterministic Seeding of KMeans for Text Document Clustering",
   booktitle="Experimental IR Meets Multilinguality, Multimodality, and Interaction",
   year="2018",
   publisher="Springer International Publishing",
   address="Cham",
   pages="76--88",
   isbn="978-3-319-98932-7",
}
```
