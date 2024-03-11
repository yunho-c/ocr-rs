#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

// basics
use std::error::Error;
use std::path::{Path, PathBuf};
// arrays/vectors/tensors
use ndarray::{array, Array, Array1, Array2, Array3, Array4, ArrayBase, IxDynImpl};
use ndarray::{s, Axis, Dim, IxDyn};
use ndarray::{ViewRepr, OwnedRepr};
// images
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, };
use image::imageops::FilterType;
// use imageproc::drawing::draw_filled_rect_mut;
// use imageproc::rect::Rect;
// machine learning
use ort::{Session, GraphOptimizationLevel};


fn read_dict(path_str: &str) -> Result<Vec<String>, Box<dyn Error>> {
    // let dict_str = std::fs::read_to_string("./assets/onnx/en/en_dict.txt")?;
    let dict_str = std::fs::read_to_string(path_str)?;
    let dict: Vec<String> = dict_str.split("\n").map(str::to_string).collect();
    // println!("{:?}", dict); // DEBUG
    Ok(dict)
}


// fn thresholded_argmax(a: Array2<f32>, threshold: f32) -> Array1<u32> { // takes argmax of rows above threshold // https://stackoverflow.com/a/57963733/15275714
fn thresholded_argmax(a: Array<f32, Dim<IxDynImpl>>, threshold: f32) -> Array1<u32> { // takes argmax of rows above threshold // https://stackoverflow.com/a/57963733/15275714
    let mut indices = Array1::zeros(a.shape()[0]);
    for (i, row) in a.axis_iter(Axis(0)).enumerate() {
        let (max_idx, max_val) =
            row.iter()
                .enumerate()
                .fold((0, row[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
            // println!("max idx for row {}: {}", i, max_idx); // DEBUG
            if max_val > threshold { indices[i] = max_idx as u32; }
            else { indices[i] = 0; }
    }
    return indices;
}

fn drop_zeros(a: Array1<u32>) -> Array1<u32> {
    let mut b: Vec<u32> = Vec::new();
    for i in 0..a.len() {
        if a[i] != 0 {
                b.push(a[i]);
            }
        }
    return Array1::from_vec(b);
}


fn image_to_onnx_input(image: DynamicImage) -> Array4<f32> {
    let mut img_arr = image.to_rgb8().into_vec();
    let (width, height) = image.dimensions();
    let channels = 3;
    let mut onnx_input = Array::zeros((1, channels, height as _, width as _));
    for (x, y, pixel) in image.into_rgb8().enumerate_pixels() {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        // Set the RGB values in the array
        onnx_input[[0, 0, y as _, x as _]] = (r as f32) / 255.;
        onnx_input[[0, 1, y as _, x as _]] = (g as f32) / 255.;
        onnx_input[[0, 2, y as _, x as _]] = (b as f32) / 255.;
      };
    onnx_input
    //   x_d = np.array(img_d).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)/256 // HWC -> NCHW
  }


struct Ocr {
    det_file: PathBuf,
    det_model: Session,
    det_size: u32,

    rec_file: PathBuf,
    rec_model: Session,
    char_list: Vec<String>,
}

impl Ocr {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let det_file = PathBuf::from("./assets/onnx/en/en_PP-OCRv3_det_infer.onnx");
        // let det_file = PathBuf::from("./assets/onnx/ch/ch_PP-OCRv4_det_infer.onnx");
        let det_model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(&det_file)?;
        let det_size = 224;
        let rec_file = PathBuf::from("./assets/onnx/en/en_PP-OCRv4_rec_infer.onnx");
        // let rec_file = PathBuf::from("./assets/onnx/ch/ch_PP-OCRv4_rec_infer.onnx");
        let rec_model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(&rec_file)?;
        
        // let char_list = read_dict().unwrap();
        let char_list = read_dict("./assets/onnx/en/recreated_dict.txt").unwrap();

        Ok( Self { det_file, det_model, det_size, rec_file, rec_model, char_list } )
    }

    pub fn process(self) -> Result<(), Box<dyn Error>> {
        
        // load image
        let mut det_image = ImageReader::open("./test_images/sparse_text.jpg")?.decode()?;
        det_image = det_image.resize_exact(self.det_size, self.det_size, FilterType::CatmullRom);
        
        // convert image into input
        let det_image_array = image_to_onnx_input(det_image);
        // println!("image array: {:?}", det_image_array.view());
        let det_inputs = ort::inputs!["x" => det_image_array.view()]?; 
        
        // run it through det model
        let det_outputs = self.det_model.run(det_inputs)?;
        let det_preds = det_outputs["sigmoid_0.tmp_0"].extract_tensor::<f32>()?;
        let det_preds_view = det_preds.view().clone().into_owned();
        // println!("det_preds_view: {:?}", det_preds_view.view());
        
        // turn preds into bboxes
        
        // create image with bboxes
        // let mut rec_image = ImageReader::open("./test_images/sparse_text_box.jpg")?.decode()?;
        // let mut rec_image = ImageReader::open("./test_images/sparse_text_box_2.jpg")?.decode()?;
        // let mut rec_image = ImageReader::open("./test_images/sparse_text_box_3.jpg")?.decode()?;
        // let mut rec_image = ImageReader::open("./test_images/sparse_text_box_4.jpg")?.decode()?;
        let mut rec_image = ImageReader::open("./test_images/punctuations_box.jpg")?.decode()?;
        // let mut rec_image = ImageReader::open("./test_images/sparse_text_box_5.jpg")?.decode()?;
        // let mut rec_image = ImageReader::open("./test_images/sparse_text_box_6.jpg")?.decode()?;
        let mut rec_image = ImageReader::open("./test_images/sparse_text_box_7.jpg")?.decode()?;
        // rec_image = rec_image.resize_exact(184, 48, FilterType::CatmullRom);
        // rec_image = rec_image.resize_exact(285, 48, FilterType::CatmullRom);
        rec_image = rec_image.resize_exact(512, 48, FilterType::CatmullRom);
        // rec_image = rec_image.resize_exact(1000, 48, FilterType::CatmullRom);

        // convert image into input
        let rec_image_array = image_to_onnx_input(rec_image);
        
        // run it through rec model
        let rec_inputs = ort::inputs!["x" => rec_image_array.view()]?; 
        let rec_outputs = self.rec_model.run(rec_inputs)?;
        // let rec_preds = rec_outputs["softmax_11.tmp_0"].extract_tensor::<f32>()?; // ch_PP-OCRv4
        let rec_preds = rec_outputs["softmax_2.tmp_0"].extract_tensor::<f32>()?; // en_PP-OCRv4
        let rec_preds_view = rec_preds.view().clone().into_owned();
        // println!("rec_preds_view: {:?}", rec_preds_view.view());

        // run NMS
        for rec_pred in rec_preds_view.axis_iter(Axis(0)) {
            // println!("rec_pred: {:?}", rec_pred); // DEBUG
                
            let indices = thresholded_argmax(rec_pred.into_owned(), 0.90);
            let nonzero_indices = drop_zeros(indices);
            println!("rec_pred (plus 1): {:?}", nonzero_indices.clone() + 1); // DEBUG

            let mut chars: Vec<String> = Vec::new();
            // convert indices into characters
            // for idx in indices.axis_iter(Axis(0)) {
            for idx in nonzero_indices.iter() {
                let char = self.char_list.get(*idx as usize).unwrap();
                // println!("char: {}", char);
                chars.push(char.to_string());
            }
            let text: String = chars.join("");
            println!("detected text: {}", text);
        }

        // output characters and locations (bboxes)



        Ok(())
    }
}

fn test() {
    let mut ocr = Ocr::new().unwrap();
    ocr.process().unwrap();

    // let rec_pred = array![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]];
    // println!("{:?}", thresholded_argmax(rec_pred, 0.5));
}

fn main() {
    // println!("Hello, world!");
    println!("{:?}", test());
}
