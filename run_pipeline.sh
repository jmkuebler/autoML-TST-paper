# MNIST

python3 pipeline.py mnist orig univ
python3 pipeline.py mnist small_gn_shift univ
python3 pipeline.py mnist medium_gn_shift univ
python3 pipeline.py mnist large_gn_shift univ
python3 pipeline.py mnist small_image_shift univ
python3 pipeline.py mnist medium_image_shift univ
python3 pipeline.py mnist large_image_shift univ
python3 pipeline.py mnist adversarial_shift univ
python3 pipeline.py mnist ko_shift univ
python3 pipeline.py mnist medium_img_shift+ko_shift univ
python3 pipeline.py mnist only_zero_shift+medium_img_shift univ



# CIFAR-10

python3 pipeline.py cifar10 orig univ
python3 pipeline.py cifar10 small_gn_shift univ
python3 pipeline.py cifar10 medium_gn_shift univ
python3 pipeline.py cifar10 large_gn_shift univ
python3 pipeline.py cifar10 small_image_shift univ
python3 pipeline.py cifar10 medium_image_shift univ
python3 pipeline.py cifar10 large_image_shift univ
python3 pipeline.py cifar10 adversarial_shift univ
python3 pipeline.py cifar10 ko_shift univ
python3 pipeline.py cifar10 medium_img_shift+ko_shift univ
python3 pipeline.py cifar10 only_zero_shift+medium_img_shift univ



