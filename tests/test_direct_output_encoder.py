import numpy as np
import pytest
from PIL import Image, JpegImagePlugin

from focus_stack_app.hugin import output_encoder as encoder


@pytest.mark.parametrize('quality,subsampling', [(100, 0), (92, 2)])
def test_direct_jpeg_matches_lossless_intermediate_and_preserves_profile(tmp_path, quality, subsampling):
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
    image.info['icc_profile'] = b'test-profile'
    source = tmp_path/'source.tif'
    image.save(source, compression='tiff_deflate', icc_profile=image.info['icc_profile'])
    cfg = encoder.OutputConfig(jpeg_quality=quality, jpeg_subsampling=subsampling)
    old = encoder.encode_output(source, tmp_path/'old.jpg', config=cfg)
    direct = encoder.encode_image_output(image, tmp_path/'direct.jpg', config=cfg)
    assert old.read_bytes() == direct.read_bytes()
    with Image.open(direct) as actual:
        assert actual.info['icc_profile'] == b'test-profile'
        assert JpegImagePlugin.get_sampling(actual) == subsampling
    assert image.getpixel((0, 0)) is not None  # caller retains image ownership


def test_direct_tiff_preserves_pixels_and_compression(tmp_path):
    pixels = np.arange(64*96, dtype=np.uint16).reshape(64, 96)
    image = Image.fromarray(pixels)
    target = encoder.encode_image_output(image, tmp_path/'out.tif', config=encoder.OutputConfig(format='tiff'))
    with Image.open(target) as result:
        np.testing.assert_array_equal(result, pixels)
        assert result.info['compression'] == 'tiff_adobe_deflate'


@pytest.mark.parametrize('failure', ['save', 'validation'])
def test_failed_encoding_retains_previous_output_and_cleans_partial(tmp_path, monkeypatch, failure):
    destination = tmp_path/'out.jpg'
    destination.write_bytes(b'existing image')
    def fail(image, path, cfg):
        path.write_bytes(b'incomplete jpeg')
        if failure == 'save':
            raise OSError('disk full')
    monkeypatch.setattr(encoder, '_save_encoded_image', fail)
    with pytest.raises(encoder.OutputEncodingError):
        encoder.encode_image_output(Image.new('RGB', (32, 24)), destination, config=encoder.OutputConfig(overwrite=True))
    assert destination.read_bytes() == b'existing image'
    assert not list(tmp_path.glob('*.partial'))


def test_concurrent_destination_is_not_clobbered(tmp_path, monkeypatch):
    destination = tmp_path/'out.jpg'
    original_save = encoder._save_encoded_image
    def publish_other(image, path, cfg):
        original_save(image, path, cfg)
        destination.write_bytes(b'other completed output')
    monkeypatch.setattr(encoder, '_save_encoded_image', publish_other)
    with pytest.raises(encoder.OutputCollisionError):
        encoder.encode_image_output(Image.new('RGB', (32, 24)), destination)
    assert destination.read_bytes() == b'other completed output'
    assert not list(tmp_path.glob('*.partial'))


def test_direct_output_protects_original_even_when_overwrite_requested(tmp_path):
    destination = tmp_path/'source.jpg'
    destination.write_bytes(b'original')
    with pytest.raises(encoder.OutputCollisionError, match='original image'):
        encoder.encode_image_output(Image.new('RGB', (32, 24)), destination,
                                    config=encoder.OutputConfig(overwrite=True), original_path=destination)
    assert destination.read_bytes() == b'original'

