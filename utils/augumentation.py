import cv2
import os
from pathlib import Path
from scipy.ndimage import rotate,shift
import random
import argparse
import os

# working Directory
workingDirectory = Path.cwd()
print(workingDirectory)

# default path 
## original data paths
folderImg = Path('Ressources\images')
folderMsk = Path('Ressources\masks')
default_images_path = workingDirectory / folderImg
default_masks_path = workingDirectory / folderMsk
## paths for saving folders for augmented data
folderAugImg = Path('Ressources\img_augmented_path')
folderAugMsk = Path('Ressources\msk_augmented_path')
default_images_aug_path = workingDirectory / folderAugImg
default_masks_aug_path = workingDirectory / folderAugMsk
default_seed = 42

class Augumentator:
    """A class for augmenting images with various transformations such as rotations, flips, and translations."""
    
    def __init__(self,
                 img_path=default_images_path,
                 msk_path=default_masks_path,
                 seed = default_seed):
        """
        Initializes the Augmentator with paths for images and masks, and a seed for randomness.

        Args:
            img_path (Path): The file path to the directory containing images.
            msk_path (Path): The file path to the directory containing masks.
            seed (int): Seed value for random number generator to ensure reproducibility.
        """
        self.img_path = Path(img_path)
        self.msk_path = Path(msk_path)
        self.images=[] 
        self.images_name=[]
        self.masks=[]
        self.masks_name=[]
        self.counter= 0
        
        self.seed = seed
        random.seed(self.seed)
        
        # load pictures and its masks
        if self.img_path.exists():
            for im in os.listdir(self.img_path):
                if im.endswith(('.jpg', '.jpeg', '.png')):
                    image = cv2.imread(str(self.img_path / im))
                    if image is not None:
                        self.images.append(image)
                        self.images_name.append(im)
                
        if self.msk_path.exists():
            for msk in os.listdir(self.msk_path):
                if msk.endswith(('.jpg', '.jpeg', '.png')):
                    mask = cv2.imread(str(self.msk_path / msk))
                    if mask is not None:
                        self.masks.append(mask)
                        self.masks_name.append(msk)

    def set_seed(self,seed):
        """Set a new seed for random number generation."""
        self.seed = seed
        random.seed(self.seed)

    def rotation(self,angle_range=(-30, 30),inplace=False):
        """Rotate images randomly within a specified angle range."""
        r_img = []
        r_msk = []
        angle = random.randint(*angle_range)
        for img in self.images:
            rotated_img = rotate(img, angle=angle, reshape=False, mode='nearest')
            r_img.append(rotated_img)
        for msk in self.masks:
            rotated_msk = rotate(msk, angle=angle, reshape=False, mode='nearest')
            r_msk.append(rotated_msk)  
        if inplace:
            self.images,self.masks = r_img,r_msk          
        return r_img,r_msk
    

    def h_flip(self,inplace=False):
        """Flip images horizontally."""
        hflipped_img = [cv2.flip(img, 1) for img in self.images]
        hflipped_msk = [cv2.flip(msk, 1) for msk in self.masks]
        if inplace:
            self.images,self.masks = hflipped_img,hflipped_msk
        return hflipped_img,hflipped_msk

    def v_flip(self,inplace=False):
        """Flip images vertically."""
        vflipped_img = [cv2.flip(img, 0) for img in self.images]
        vflipped_msk = [cv2.flip(msk, 0) for msk in self.masks]
        if inplace:
            self.images,self.masks = vflipped_img,vflipped_msk
        return vflipped_img,vflipped_msk

    def h_transl(self, pixel_range=(-20, 20),inplace=False):
        """Translate images horizontally within a specified pixel range. default values are -20 to 20"""
        htranslated_img = []
        htranslated_msk = []
        pixels = random.randint(*pixel_range)
        for img in self.images:
            translated_img = shift(img, [0, pixels, 0])
            htranslated_img.append(translated_img)
        for msk in self.masks:
            translated_msk = shift(msk, [0, pixels, 0])
            htranslated_msk.append(translated_msk)
        if inplace:
            self.images,self.masks = htranslated_img,htranslated_msk
        return htranslated_img,htranslated_msk

    def v_transl(self, pixel_range=(-20, 20),inplace=False):
        """Translate images vertically within a specified pixel range. default values are -20 to 20"""
        vtranslated_img = []
        vtranslated_msk = []
        pixels = random.randint(*pixel_range)
        for img in self.images:
            translated_img = shift(img, [pixels, 0, 0])
            vtranslated_img.append(translated_img)
        for msk in self.masks:
            translated_msk = shift(msk, [0, pixels, 0])
            vtranslated_msk.append(translated_msk)
        if inplace:
            self.images,self.masks = vtranslated_img,vtranslated_msk
        return vtranslated_img,vtranslated_msk
      
    def save(self,
              img_path=default_images_aug_path,
              mask_path=default_masks_aug_path,
              save_in_tiff=False):
        """Save the augmented images and masks in the specified directory."""
        
        
        """
        # old codes
        if not save_in_tiff:
            for id in range(len(self.masks_name)):
                out_path = os.path.join(mask_path, self.msk_name[id])
                cv2.imwrite(out_path, self.masks[id])
            for id in range(len(self.images_name_name)):
                out_path = os.path.join(img_path, self.images_name_name[id])
                cv2.imwrite(out_path, self.images[id])
        else:
            for id in range(len(self.masks_name)):
                last_dot_index = self.msk_name[id].rfind(".")
                out_path = os.path.join(mask_path, self.msk_name[id][:last_dot_index]+".tiff")
                cv2.imwrite(out_path, self.masks[id])
            for id in range(len(self.images_name_name)):
                last_dot_index = self.images_name_name[id].rfind(".")
                out_path = os.path.join(mask_path, self.images_name_name[id][:last_dot_index]+".tiff")
                cv2.imwrite(out_path, self.images[id])
        """
        self.counter += 1
        img_extension = ".tiff" if save_in_tiff else ".png"
        if not os.path.exists(Path(img_path)):
            os.makedirs(img_path)
        if not os.path.exists(Path(mask_path)):  
            os.makedirs(mask_path)
        
        for id, img in enumerate(self.images):
            img_filename = "aug_" +str(self.counter) +"_"+ self.images_name[id].rsplit('.', 1)[0] + img_extension
            cv2.imwrite(str(Path(img_path) / img_filename), img)
        for id, msk in enumerate(self.masks):
            msk_filename = "aug_" +str(self.counter) +"_"+self.masks_name[id].rsplit('.', 1)[0] + img_extension
            cv2.imwrite(str(Path(mask_path) / msk_filename), msk)






                
def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Augment images with specific transformations.")
    parser.add_argument("--img_path", type=str, default="Ressources/images", help="Directory path to the image files.")
    parser.add_argument("--msk_path", type=str, default="Ressources/masks", help="Directory path to the mask files.")
    parser.add_argument("--img_aug_path", type=str, default="Ressources/img_augmented_path", help="Directory path for saving augmented images.")
    parser.add_argument("--msk_aug_path", type=str, default="Ressources/msk_augmented_path", help="Directory path for saving augmented masks.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")
    parser.add_argument("--process", type=str, nargs='+', choices=['rotation', 'h_flip', 'v_flip', 'h_transl', 'v_transl'],
                        help="List of augmentation processes to apply: rotation, h_flip, v_flip, h_transl, v_transl.")
    return parser.parse_args()
def main():
    args = parse_args()
    augmenter = Augumentator(args.img_path, args.msk_path, args.img_aug_path, args.msk_aug_path, args.seed)
    
    # Process selected augmentations
    if 'rotation' in args.process:
        rotated_images = augmenter.rotation()
    if 'h_flip' in args.process:
        h_flipped_images = augmenter.h_flip()
    if 'v_flip' in args.process:
        v_flipped_images = augmenter.v_flip()
    if 'h_transl' in args.process:
        h_translated_images = augmenter.h_transl()
    if 'v_transl' in args.process:
        v_translated_images = augmenter.v_transl()
    
    augmenter.save()

if __name__ == "__main__":
    main()