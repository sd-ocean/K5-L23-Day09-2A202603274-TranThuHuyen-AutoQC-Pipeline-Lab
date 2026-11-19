from pipeline.checks import aspect_ratio


def test_extreme_pixel_aspect_ratio_warns_with_original_line(make_ds, cfg):
    ds = make_ds(images={"wide": 1}, labels={"wide": ["0 .5 .5 .8 .02"]}, size=(320, 240))
    issues = aspect_ratio.run(ds, cfg)
    assert [(i.code, i.image, i.line, i.severity) for i in issues] == [
        ("BOX_EXTREME_ASPECT", "wide", 1, "warn")
    ]
    assert issues[0].extra["aspect_ratio"] > 10


def test_reasonable_aspect_ratio_is_silent_and_height_is_accounted_for(make_ds, cfg):
    ds = make_ds(images={"normal": 1}, labels={"normal": ["0 .5 .5 .4 .4"]}, size=(640, 480))
    assert aspect_ratio.run(ds, cfg) == []


def test_invalid_labels_are_skipped_without_crashing(make_ds, cfg):
    ds = make_ds(images={"bad": 1}, labels={"bad": ["garbage", "93 .5 .5 .8 .02", "0 .5 .5 .8 0"]})
    assert aspect_ratio.run(ds, cfg) == []
