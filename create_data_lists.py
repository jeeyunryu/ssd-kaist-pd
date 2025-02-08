from codes_for_kaist.utils_fusion import create_data_lists

if __name__ == '__main__':
    # create_data_lists(voc07_path='../ssd/datasets/pascal_voc/VOC2007',
    #                   voc12_path='../ssd/datasets/pascal_voc/VOC2012',
    #                   output_folder='./output_folder')
    create_data_lists('/home/urp4/workspace/src/ssd/datasets/kaist', output_folder='./')
