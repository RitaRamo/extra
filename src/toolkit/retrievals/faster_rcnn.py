import faiss
import numpy as np
import os

class ImageRetrieval():

    def __init__(self, retrieval_dir, dim_examples, train_dataloader_images, device, is_to_add=False, conceptual_captions=False, index_name=None):

        self.device=device
        self.retrieval_dir=retrieval_dir
        self.index_name = index_name

        if is_to_add:
            print("adding")
            self.datastore = faiss.IndexIDMap(faiss.IndexFlatIP(dim_examples)) #datastore
            self._add_examples(train_dataloader_images)
        else:
            print("loading")
            self.datastore = faiss.read_index(os.path.join(retrieval_dir, index_name))
                


    def _add_examples(self, train_dataloader_images):
        print("\nadding input examples to datastore (retrieval)")
        for i, (encoder_output, imgs_indexes) in enumerate(train_dataloader_images):
            input_img = encoder_output.mean(dim=1)
            input_img = input_img / input_img.norm(dim=-1, keepdim=True)            
            self.datastore.add_with_ids(input_img.cpu().numpy(), np.array(imgs_indexes, dtype=np.int64))            

            if i%5==0:
                print("i and img index of ImageRetrival",i, imgs_indexes)
                print("n of examples", self.datastore.ntotal)

        faiss.write_index(self.datastore,os.path.join(self.retrieval_dir, self.index_name))
    
    def retrieve_nearest_for_train_query(self, query_img):
        #get nearest image
        k=2 #the first image is itself, so use the second image as nearest
        D, I = self.datastore.search(query_img, k)
        return I[:,1]

    def retrieve_nearest_for_val_or_test_query(self, query_img):
        #get nearest image
        k=1
        D, I = self.datastore.search(query_img, k) 
        return I[:,0]

    def retrieve_nearest_for_train_query_with_D(self, query_img):
        #get nearest image
        k=2
        D, I = self.datastore.search(query_img, k) 
        return D[:,1], I[:,1]

    def retrieve_nearest_for_train_query_with_D_given_k(self, query_img):
        #get nearest image
        k=2
        D, I = self.datastore.search(query_img, k) 
        return D[:,1], I[:,1]


    def retrieve_nearest_for_train_query_with_D(self, query_img,  k=1):
        #get nearest image
        D, I = self.datastore.search(query_img, k) 
        return D[:,1:k], I[:,1:k]

    def retrieve_nearestk_for_train_query(self, query_img, k=1):
        #get k nearest image
        k=k+1 #the first image is itself, so use the nexts one as nearest
        D, I = self.datastore.search(query_img, k)
        return I[:,1:k]

    def retrieve_nearestk_for_val_or_test_query(self, query_img, k=1):
        #get k nearest image
        D, I = self.datastore.search(query_img, k)
        return I[:, :k]
