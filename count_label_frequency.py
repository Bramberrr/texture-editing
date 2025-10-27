import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

def show_label_frequency(label_json="captions_labels.json", top_k=50):
    """
    Load label dictionary and show frequency of each label across all images.
    Also saves a bar chart of top labels as label_frequency.png
    """
    # --- Load data ---
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Flatten all labels ---
    all_labels = []
    for fname, labels in data.items():
        if isinstance(labels, list):
            all_labels.extend(labels)

    # --- Count frequencies ---
    counter = Counter(all_labels)
    total_labels = len(counter)

    print(f"Total unique labels: {total_labels}")
    print("\nTop labels:")
    for label, count in counter.most_common(top_k):
        print(f"{label:20s} {count}")

    # --- Optional visualization ---
    top_labels = dict(counter.most_common(top_k))
    plt.figure(figsize=(10, 6))
    plt.barh(list(top_labels.keys())[::-1], list(top_labels.values())[::-1], color='skyblue')
    plt.xlabel("Frequency")
    plt.title(f"Top {top_k} Labels in Dataset")
    plt.tight_layout()
    plt.savefig("label_frequency.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    show_label_frequency("captions_labels.json", top_k=200)


BS_wins = [["grass","flower","plant","leave","leaf"],["raspberry","pea","lemon"]]

CLIP_wins = [["wall","stone","brick","cobblestone","pebblestone","roof","building","street","block"],["wooden","wood","fine"],["fabric","cloth","material","blanket"],
        ["concrete","ground","granite","gravel","dirt","sand","beach","pavement","road","snow","substance","mud","dry"],
        ["marble","speckled"],["rock"],["carpet","curtain"],["tile"],["paper","cardboard"],["metal","gold"],["water","droplet","pool","wave"],["skin","leather"],
        ["fence"],["moss"],["tree","bark","forest"],["bean","food","nut","pistachio","corn","peanut","berry","almond","rice"],["basket","weave","woven","mesh","net","link"],
        ["rust","rusty","rusted"],["shaving"],["wool","hair","fur","furry","yarn"],["grape","apple","cherry"," tomato","pepper","lime"],
        ["blueberry","strawberry","cranberry","blackberry","pomegranate","pear","banana"],["candy","sugar","crystal","bead"]]

no_gloss = [["striped","stripe"],["checkerboard"],["camouflage","abstract"],["polka","dot"],["glitter","star"],["bamboo"],["screen"]]
"""

Top labels:
background           1207
white                954
pattern              936
wall                 900
black                554
fabric               486
stone                463
brick                448
green                430
blue                 410
wooden               404
concrete             390
wood                 374
brown                335
red                  317
floor                270
grain                258
ground               245
diffuse              240
marble               237
gray                 226
rock                 197
pink                 194
yellow               189
light                178
bunch                172
carpet               171
tile                 171
paper                170
pile                 169
metal                164
dirt                 153
dark                 149
water                147
purple               129
skin                 127
line                 124
field                121
grass                119
fence                118
leather              113
old                  113
sand                 111
dot                  107
granite              107
different            101
person               94
square               93
spot                 92
gravel               90
polished             88
paint                85
small                84
color                80
unsplash             79
orange               78
papi                 77
piece                73
abstract             71
moss                 70
striped              70
many                 68
bright               65
checkerboard         64
plaster              63
cobblestone          62
beige                61
body                 58
tree                 56
beach                55
blurry               54
pavement             54
hole                 52
leave                51
sheet                50
polka                50
road                 49
colored              48
seamless             48
sky                  47
middle               47
stripe               47
large                46
cloth                46
diamond              44
side                 43
speckled             39
roof                 39
material             38
bamboo               37
rough                36
digital              36
flower               34
plant                34
camouflage           33
star                 32
top                  32
border               32
bark                 31
snow                 30
vintage              30
building             29
blanket              27
bean                 27
street               27
basket               27
block                27
grey                 27
rust                 27
textile              27
weave                26
bubble               26
gold                 25
substance            25
group                25
plank                25
woven                24
chair                23
slat                 23
hexagonal            23
vertical             23
fine                 23
droplet              22
night                21
colorful             21
silver               21
rusty                21
painting             21
hintergrund          21
wallpaper            21
people               20
pexel                20
window               19
cement               19
other                19
heart                19
circle               19
pool                 19
glitter              19
forest               19
screen               18
cloud                18
wave                 18
rusted               18
shade                17
cardboard            16
asphalt              16
mesh                 15
center               15
table                15
ceramic              15
leaf                 15
food                 14
shape                14
grate                14
pebblestone          14
design               14
shaving              14
semi                 14
precious             14
sidewalk             14
animal               13
crack                13
mud                  13
dry                  13
individual           13
rumee                13
print                12
seed                 12
computer             12
base                 12
jpg                  12
room                 12
chain                12
curtain              12
board                12
tamanna              12
nut                  11
dune                 11
burlap               11
wool                 11
denim                11
wicker               11
plate                11
walkway              11
olive                11
trunk                11
shiny                11
bead                 11
bead                 11
bead                 11
link                 11
bead                 11
link                 11
bead                 11
link                 11
bead                 11
link                 11
bead                 11
link                 11
bead                 11
link                 11
bead                 11
bead                 11
bead                 11
bead                 11
bead                 11
link                 11
glass                11
face                 11
yarn                 10
aerial               10
decorative           10
parquet              10
stain                10
metallic             10
wavy                 10
turquoise            10
"""