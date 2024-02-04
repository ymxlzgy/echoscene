# SG-FRONT

SG-FRONT provides semantic scene graphs for [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) (refined version). A semantic scene graph is defined by a set of tuples between nodes and edges where nodes represent specific 3D object instances. Nodes are defined by its semantics, a hierarchy of classes. The edges in our graphs are the semantic relationships (predicates) between the nodes, including `standing on` or `same material as`, following [3DSSG](https://github.com/3DSSG/3DSSG.github.io).

## Data Organization
Assume you have downloaded [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) and preprocess the data following <a href="https://github.com/nv-tlabs/ATISS#data-preprocessing">ATISS</a>.

**Download link:** [SG-FRONT](https://www.campar.in.tum.de/public_datasets/2023_commonscenes_zhai/SG_FRONT.zip) and [additional bboxes](https://www.campar.in.tum.de/public_datasets/2023_commonscenes_zhai/bbox.zip). Download and unzip them to `path/to/3D-FRONT`. The data in the provided files is organized as follows:
```
SG-FRONT.zip
|-- relationships_{type}_trainval.json
    List of all triplets consisting of semantic classes (nodes) and semantic relationships (edges) in the trainval split.
|-- relationships_{type}_test.json
    List of all triplets consisting of semantic classes (nodes) and semantic relationships (edges) in the test split.
|-- classes_{type}.txt
    List of the semantic classes in the current room type.
|-- relationships.txt
    List of all relationship types.

bbox.zip
|-- obj_boxes_{type}_trainval.json
    Oriented bounding boxes in each scene in the trainval split.
|-- obj_boxes_{type}_test.json
    Oriented bounding boxes in each scene in the test split.
|-- boxes_centered_stats_{}_trainval.txt
	Box normalization parameters in the trainval split.
|-- boxes_centered_stats_{}_test.txt
	Box normalization parameters in the test split.
|-- cat_jid_*.json (not necessary for commonscenes)
	Object instance id with the bounding box size.
```

## Data Formats

SG-FRONT is for the general purpose of scene graph research:

**Object relationships (edges) (`relationships_*.json`)**:

```javascript
{
  "scans": [
    {
      "scan": "MasterBedroom-33296",
      "objects": {"1": "dressing_table", "2": "nightstand", "3": "stool", "4": "double_bed", "5": "cabinet", "6": "pendant_lamp", "7": "wardrobe", "8": "floor"},
      "relationships": [
        [
          6,  // subject instance id in the current scene
          5,  // object instance id in the current scene
          1, // relationships id (order aligned with relationships.txt)
          "left" // textual semantic relationship (from relationships.txt)
        ],
        [ 5, 2, 14, "same super category as" ],
        [ 4, 1, 10, "taller than" ],
        [ 7, 2, 15, "same material as" ], 
        ...
      ],
      
    }, { 
      ... 
    }
  ]
}
```

Bbox information is for additional usages:

**Object bboxes (`obj_boxes_*.json`)**:

```javascript
{
  "MasterBedroom-33296": { // room id from preprocessed 3D-FRONT
      "1": {"param7": [1.0406779999999998, 1.098808, 0.4491219999999996, -5.370314764104733, 0.0, -4.066067038275899, 0], // bbox size and pose (x,y,z,angle), with positive y-axis up.
          	"8points": [[-5.890653764104733, 0.0, -4.290628038275899], [-5.890653764104733, 0.0, -3.841506038275899], [-5.890653764104733, 1.098808, -4.290628038275899], [-5.890653764104733, 1.098808, -3.841506038275899], [-4.849975764104733, 0.0, -4.290628038275899], [-4.849975764104733, 0.0, -3.841506038275899], [-4.849975764104733, 1.098808, -4.290628038275899], [-4.849975764104733, 1.098808, -3.841506038275899]], // bbox vertices
            "scale": [1, 1, 1], // always 1
            "model_path": "/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FUTURE-model/1791e0c9-15fc-4fa9-80df-e73332ed6ce0/raw_model.obj"}, // the object model in 3D-FUTURE. The head path needs to be changed.
      "2": {"param7": [0.48633499999999996, 0.607177, 0.4845579999999998, -2.560559506120386, 0.0, -4.061041670654544, 0],
         	"8points": [[-2.803745506120386, 0.0, -4.303502670654543], [-2.803745506120386, 0.0, -3.8189446706545436], [-2.803745506120386, 0.607177, -4.303502670654543], [-2.803745506120386, 0.607177, -3.8189446706545436], [-2.3174105061203862, 0.0, -4.303502670654543], [-2.3174105061203862, 0.0, -3.8189446706545436], [-2.3174105061203862, 0.607177, -4.303502670654543], [-2.3174105061203862, 0.607177, -3.8189446706545436]],
            "scale": [1, 1, 1], 
            "model_path": "/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FUTURE-model/62319141-32fa-4353-b2ff-ec31ca232e3e/raw_model.obj"}, 
      ...,
      "scene_center": [-2.9265, 0.0, -2.7]}, // the average center of the current room
  
  "MasterBedroom-9404": {
      ...,
      "scene_center": [2.96192, 0.0, 0.32996000000000003]}
  
  ...
}
```

### Notes:

* Our automatic labeling setup can be downloaded [here](https://www.campar.in.tum.de/public_datasets/2023_commonscenes_zhai/script.zip) for the reference and the potential adjustment. This can **ONLY** be run after preprocessing 3D-FRONT following <a href="https://github.com/tangjiapeng/DiffuScene#pickle-the-3d-future-dataset">Diffuscene</a> or  <a href="https://github.com/nv-tlabs/ATISS#data-preprocessing">ATISS</a>.
