import cv2
import numpy as np
import time



if __name__ == "__main__":
    image = cv2.imread('test1.jpg', cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        import pandas as pd
        # Для отображения выберем небольшой фрагмент (например, 10x10 пикселей из начала, т.е. из верхнего левого угла изображения), 
        # чтобы не загромождать консоль
        h, w = image.shape
        print(f"Размер изображения: {w}x{h}")

        print(type(image))           # <class 'numpy.ndarray'>
        print(image.shape)           # (высота, ширина) - 2 измерения
        print(image.dtype)           # uint8
        print(isinstance(image, np.ndarray))  # True
        
        # Создаем DataFrame из фрагмента изображения
        sample_size = 10
        df = pd.DataFrame(image[:sample_size, :sample_size])
        
        print(f"\nТаблица пикселей (фрагмент {sample_size}x{sample_size}):")
        print(df.to_string())

        t0 = time.perf_counter()
        img = cv2.add(image, 50)
        t1 = time.perf_counter()

        df2 = pd.DataFrame(img[:sample_size, :sample_size])
        
        print(f"\nТаблица пикселей после добавления 50 к каждому пикселю (фрагмент {sample_size}x{sample_size}):")
        print(df2.to_string())

        print(f"Время сложения с помошью cv2.add: {t1 - t0:.5f} s") 

        # =============================================================

        t2 = time.perf_counter()
        new_image = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                res = int(image[i, j]) + 50
                if res > 255:
                    res = 255
                new_image[i, j] = res
        t3 = time.perf_counter()


        df3 = pd.DataFrame(new_image[:sample_size, :sample_size])
        
        print(f"\nТаблица пикселей после добавления 50 к каждому пикселю (фрагмент {sample_size}x{sample_size}):")
        print(df3.to_string())

        print(f"Время сложения с циклом for: {t3 - t2:.5f} s") 


        
        #cv2.imshow('logo', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        print("Ошибка: не удалось загрузить 'test1.jpg'")