postprocessing의 한 종류.

<img src="https://github.com/YeoungJun0508/study_oml/assets/145903037/0951558d-70f4-4676-b02c-ecc5800cf522" width="500" height="500">


Query를 oml실행 시켜서 Result를 얻고 상위 n개의 Result와 Query를 Dotproduct하여 Query로 만든 후 oml을 실행시킨다.

(더 좋은 성능)

상위 3개를 re-ranking하여 결과에 반영하는 코드

```
extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)
transform = get_normalisation_resize_torch(im_size=64)

embeddings_train, embeddings_val, df_train, df_val = \
    inference_on_dataframe(dataset_root, "/content/drive/MyDrive/Colab Notebooks/OML/picture/obj123.csv", extractor=extractor, transforms=transform)

# We are building Siamese model on top of existing weights and train it to recognize positive/negative pairs
siamese = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])
optimizer = torch.optim.SGD(siamese.parameters(), lr=1e-6)
miner = PairsMiner(hard_mining=True)
criterion = BCEWithLogitsLoss()

train_dataset = DatasetWithLabels(df=df_train, transform=transform, extra_data={"embeddings": embeddings_train})
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

for batch in train_loader:
    # We sample pairs on which the original model struggled most
    ids1, ids2, is_negative_pair = miner.sample(features=batch["embeddings"], labels=batch["labels"])
    probs = siamese(x1=batch["input_tensors"][ids1], x2=batch["input_tensors"][ids2])
    loss = criterion(probs, is_negative_pair.float())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Siamese re-ranks top-n retrieval outputs of the original model performing inference on pairs (query, output_i)
val_dataset = DatasetQueryGallery(df=df_val, extra_data={"embeddings": embeddings_val}, transform=transform)
valid_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

postprocessor = PairwiseImagesPostprocessor(top_n=3, pairwise_model=siamese, transforms=transform)
calculator = EmbeddingMetrics(postprocessor=postprocessor)
calculator.setup(num_samples=len(val_dataset))

for batch in valid_loader:
    calculator.update_data(data_dict=batch)

pprint(calculator.compute_metrics())

```


![image](https://github.com/YeoungJun0508/study_oml/assets/145903037/59aaa6f6-bed8-49fd-a23d-b88de338f9c3)



![image](https://github.com/YeoungJun0508/study_oml/assets/145903037/36c85a25-2f1e-4063-867b-7dacf0e6ee0b)



모델에 반영하는 코드

```
from oml.const import PATHS_COLUMN
from oml.registry.transforms import get_transforms_for_pretrained
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from oml.utils.misc_torch import pairwise_dist

# 1. Let's use feature extractor to get predictions
extractor = ViTExtractor.from_pretrained("vits16_dino")
transforms, _ = get_transforms_for_pretrained("vits16_dino")

_, emb_val, _, df_val = inference_on_dataframe(dataset_root, "/content/drive/MyDrive/Colab Notebooks/OML/picture/obj123.csv", extractor, transforms=transforms)

is_query = df_val["is_query"].astype('bool').values
distances = pairwise_dist(x1=emb_val[is_query], x2=emb_val[~is_query])

print("\nOriginal predictions:\n", torch.topk(distances, dim=1, k=3, largest=False)[1])

# 2. Let's initialise a random pairwise postprocessor to perform re-ranking
siamese = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])  # Note! Replace it with your trained postprocessor
postprocessor = PairwiseImagesPostprocessor(top_n=3, pairwise_model=siamese, transforms=transforms)

dataset = DatasetQueryGallery(df_val, extra_data={"embeddings": emb_val}, transform=transforms)
loader = DataLoader(dataset, batch_size=4)

query_paths = df_val[PATHS_COLUMN][is_query].values
gallery_paths = df_val[PATHS_COLUMN][~is_query].values
distances_upd = postprocessor.process(distances=distances, queries=query_paths, galleries=gallery_paths)

print("\nPredictions after postprocessing:\n", torch.topk(distances_upd, dim=1, k=3, largest=False)[1])
```

![image](https://github.com/YeoungJun0508/study_oml/assets/145903037/1e2c9911-d862-4a0c-ae25-8a18bd96e3f6)


![image](https://github.com/YeoungJun0508/study_oml/assets/145903037/0d09d863-3a7a-40af-b1e0-0fff6f0e3cc3)


