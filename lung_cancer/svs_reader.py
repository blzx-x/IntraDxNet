import numpy as np
import cv2
from openslide import OpenSlide
from loguru import logger
import traceback


class WsiReader(object):
  
  def __init__(self):
    self.reader = None
    self.read_scale = 0
    self.info = dict()
    self.clear()
    return
  
  def ReadInfo(self, filename, scale=0, _ = None):
    try:
      r = OpenSlide(filename)
      r.tile_size = int(r.properties.get("openslide.level[0].tile-height", "256"))
      r.output_tile_size = 256
      r.remap_dimensions = list(map(lambda f: np.ceil(f / r.tile_size * r.output_tile_size), r.dimensions))
    except:
      logger.warning("Error occured when loading '{}'".format(filename))
      logger.info("= = =\n{}= = =".format(traceback.format_exc()))
      return False
    
    self.reader = r
    self.read_scale = scale
    return True

  def ReadThumbnialImage(self):
    return self.pil2img(self.reader.associated_images.get("thumbnail", None))
  
  def ReadLabelImage(self):
    return self.pil2img(self.reader.associated_images.get("label", None))

  def ReadPreviewImage(self):
    return self.pil2img(self.reader.associated_images.get("macro", None))
  
  @classmethod
  def pil2img(cls, pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGBA2BGR)

  def ReadRoi(self, x, y, w, h, scale=0):
    source = self.reader
    block_x, block_y, block_w, block_h = x, y, w, h
    scan_scale = self.getScanScale()

    scale, block_x, block_y, block_w, block_h = list(map(lambda f: float(f), [scale, block_x, block_y, block_w, block_h]))
    block_w, block_h = list(map(int, [block_w, block_h]))
    real_downsample = round(scan_scale / scale, 2)
    virtual_level = source.get_best_level_for_downsample(real_downsample)
    virtual_downsample = round(source.level_downsamples[virtual_level], 2)
    downsample_frac = round(real_downsample / virtual_downsample, 2)
    virtual_scale = round(scale * downsample_frac, 2)
    # logger.debug("Match scale: {} at level: {}, virtal scale: {}, remap downsample: {} -> {}".format(
    #   scale, virtual_level, virtual_scale, real_downsample, virtual_downsample))
    virtual_location = list(map(lambda f: int(f * downsample_frac), [block_x, block_y]))
    virtual_size = list(map(lambda f: int(f * downsample_frac), (block_w, block_h)))
    # logger.debug("Read region: {} at level: {}, downsample: {}".format((*virtual_location, *virtual_size), virtual_level, virtual_downsample))
    virtual_location = list(map(lambda f: int(f * real_downsample), [block_x, block_y]))
    # logger.debug("OpenSLide.read_region acquires location in level=0, remap location to: {}".format(virtual_location))
    if virtual_location[0] > source.dimensions[0] or virtual_location[1] > source.dimensions[1]:
      logger.warning("= = = = = = = = = = = = = = Getting out-boundary image, return white picture = = = = = = = = = = = = = =")
      return np.zeros([block_h, block_w, 3], dtype="uint8") + 255
    
    roi = source.read_region(virtual_location, virtual_level, virtual_size)
    roi = roi.resize((block_w, block_h))
    
    img = self.pil2img(roi)
    return img
  
  def getWidth(self):
    w = self.getSrcWidth()
    return int(w * (self.getReadScale() / self.getScanScale()))

  def getHeight(self):
    h = self.getSrcHeight()
    return int(h * (self.getReadScale() / self.getScanScale()))
  
  def getSrcWidth(self):
    return self.reader.dimensions[0]
  
  def getSrcHeight(self):
    return self.reader.dimensions[1]
  
  def getScanScale(self):
    return int(self.reader.properties.get("aperio.AppMag", "-1"))
  
  def getReadScale(self):
    return self.read_scale or self.getScanScale()

  def setReadScale(self, scale):
    self.read_scale = scale
    return

  def clear(self):
    self.reader = None
    self.read_scale = 0
    self.info = dict()
    return
