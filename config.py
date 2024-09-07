def config():
    ref = 11.6
    ref2 = ref * 2
    ref3 = ref * 3
    ref4 = ref * 4
    return {
        # image size -> acc
        's720':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 720,
        },
        's360':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 360,
        },
        's256':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 256,
        },
        's128':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 128,
        },
        's64':{
            'sigma_points': [ref3, ref3, ref3],
            'sigma_links': [ref3, ref3, ref3],
            'img_size': 64,
        },

        # sigma size -> acc
        # update img_size after get result of img
        '1x': {
            'sigma_points': [ref *1, ref*1, ref*1],
            'sigma_links': [ref *1, ref*1, ref*1],
            'img_size': 720,
        },
        '2x': {
            'sigma_points': [ref *2, ref*2, ref*2],
            'sigma_links': [ref *2, ref*2, ref*2],
            'img_size': 720,
        },
        '3x': {
            'sigma_points': [ref *3, ref*3, ref*3],
            'sigma_links': [ref *3, ref*3, ref*3],
            'img_size': 720,
        },
        '4x': {
            'sigma_points': [ref *4, ref*4, ref*4],
            'sigma_links': [ref *4, ref*4, ref*4],
            'img_size': 720,
        },
        '5x': {
            'sigma_points': [ref *5, ref*5, ref*5],
            'sigma_links': [ref *5, ref*5, ref*5],
            'img_size': 720,
        },
        '6x': {
            'sigma_points': [ref *6, ref*6, ref*6],
            'sigma_links': [ref *6, ref*6, ref*6],
            'img_size': 720,
        },
        '10x': {
            'sigma_points': [ref *10, ref*10, ref*10],
            'sigma_links': [ref *10, ref*10, ref*10],
            'img_size': 720,
        },

        # curriculum training -> acc
        # update img_size after get result of img

        # ++   + ref2
        # ref4 - --
        '8p': {
            'sigma_points': [ref4, ref4*.9, ref4*.8],
            'sigma_links': [ref4, ref4*.9, ref4*.8],
            'img_size': 360,
        },
        '6p': {
            'sigma_points': [ref4, ref4*.8, ref4*.6],
            'sigma_links': [ref4, ref4*.8, ref4*.6],
            'img_size': 360,
        },
        '4p': {
            'sigma_points': [ref4, ref4*.7, ref4*.4],
            'sigma_links': [ref4, ref4*.7, ref4*.4],
            'img_size': 360,
        },
        '2p': {
            'sigma_points': [ref4, ref4*.6, ref4*.2],
            'sigma_links': [ref4, ref4*.6, ref4*.2],
            'img_size': 360,
        },

    }
