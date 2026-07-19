from typing import Optional, List


class ModelFullInfo:
    def __init__(self,
                 model_id: str,
                 file_path: Optional[str] = None,
                 model_url: Optional[str] = None,
                 has_code: Optional[bool] = None,
                 Size: Optional[str] = None,
                 SizeRange: Optional[str] = None,
                 input_modalities: Optional[str] = None,
                 Text_I: Optional[str] = None,
                 Image_I: Optional[str] = None,
                 Audio_I: Optional[str] = None,
                 Video_I: Optional[str] = None,
                 output_modalities: Optional[str] = None,
                 Text_O: Optional[str] = None,
                 Image_O: Optional[str] = None,
                 Audio_O: Optional[str] = None,
                 Video_O: Optional[str] = None,
                 three_d_O: Optional[str] = None,
                 model_size: Optional[str] = None,
                 input_tokens: Optional[int] = None,
                 output_tokens: Optional[int] = None,
                 downloads: Optional[int] = None,
                 likes: Optional[int] = None,
                 SizeB: Optional[int] = None,
                 code: Optional[str] = None,
                 sorted_tags: Optional[List] = None):
        if not model_id:
            raise ValueError("model_id is mandatory and cannot be None")

        self.model_id = model_id
        self.file_path = file_path
        self.model_url = model_url
        self.has_code = has_code
        self.Size = Size
        self.SizeRange = SizeRange
        self.input_modalities = input_modalities
        self.Text_I = Text_I
        self.Image_I = Image_I
        self.Audio_I = Audio_I
        self.Video_I = Video_I
        self.output_modalities = output_modalities
        self.Text_O = Text_O
        self.Image_O = Image_O
        self.Audio_O = Audio_O
        self.Video_O = Video_O
        self.three_d_O = three_d_O
        self.model_size = model_size
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.downloads = downloads
        self.likes = likes
        self.SizeB = SizeB
        self.code = code
        self.sorted_tags = sorted_tags