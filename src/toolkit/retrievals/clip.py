import faiss
import numpy as np
import os

class ClipRetrieval():

    def __init__(self, retrieval_dir, dim_examples, train_dataloader_images, device, is_to_add=False, index_name="clip_retrieval_vit"):

        self.device=device
        self.retrieval_dir=retrieval_dir
        self.index_name = index_name


        if is_to_add:
            print("creating retrieval (addding vectors to index)")
            self.datastore = faiss.IndexIDMap(faiss.IndexFlatIP(dim_examples)) #datastore
            self._add_examples(train_dataloader_images)
        else:
            print("loading the precomputed retrieval")
            self.datastore = faiss.read_index(os.path.join(retrieval_dir, index_name)) #"/media/rprstorage4/

    def _add_examples(self, train_dataloader_images):
        print("\nadding input examples to index/datastore")
        for i, (caption_embedding, imgs_indexes) in enumerate(train_dataloader_images):
            print("images fa", caption_embedding.shape)
            print("images fa", np.array(imgs_indexes.view(-1), dtype=np.int64).shape)
            self.datastore.add_with_ids(caption_embedding[0].cpu().numpy(), np.array(imgs_indexes.view(-1), dtype=np.int64))

            if i%100==0:
                print("i and img index of ImageRetrival",i, imgs_indexes)
                print("n of examples", self.datastore.ntotal)

        faiss.write_index(self.datastore, os.path.join(self.retrieval_dir, self.index_name))

    
    def retrieve_nearestk_for_train_query_with_D(self, query_img, k=10):
        #get k nearest image
        D, I = self.datastore.search(query_img, k)
        return D, I[:,:k]
