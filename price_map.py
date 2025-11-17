DISPLAY_NAMES = {
    'banh_chuoi': 'Banh chuoi',
    'banh_cua': 'Banh cua',
    'banh_mi_dua': 'Banh mi dua',
    'cha_bong': 'Banh cha bong',
    'cookie_dua': 'Cookie dua',
    'croissant': 'Croissant',
    'da_lon': 'Banh da lon',
    'egg_tart': 'Tart trung',
    'muffin': 'Muffin',
    'patechaud': 'Patechaud',
}

CAKE_PRICES_MAP = {
    'Banh chuoi': 15000,
    'Banh cua': 22000,
    'Banh mi dua': 18000,
    'Banh cha bong': 20000,
    'Cookie dua': 12000,
    'Croissant': 25000,
    'Banh da lon': 10000,
    'Tart trung': 15000,
    'Muffin': 18000,
    'Patechaud': 20000,
}

CLASSIFY_CLASSES = [
    'banh_chuoi',
    'banh_cua',
    'banh_mi_dua',
    'cha_bong',
    'cookie_dua',
    'croissant',
    'da_lon',
    'egg_tart',
    'muffin',
    'patechaud'
]

def format_currency(value):
    return f"{value:,.0f} VND"