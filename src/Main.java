import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.util.function.UnaryOperator;
/**********************************************************************************************************************/
public class Main {
    /******************************************************************************************************************/
    public static void main(String[] args) throws IOException {
        // dots();
        digits();
    }

    /******************************************************************************************************************/
    private static void dots() {
        new Thread(new FormDots()).start();
    }

    /******************************************************************************************************************/
    private static void digits() throws IOException {
        UnaryOperator<Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
        UnaryOperator<Double> dsigmoid = y -> y * (1 - y);

        NeuralNetwork neuralNetwork = new NeuralNetwork(0.001, sigmoid, dsigmoid, 784, 512, 128, 32, 10);

        /* Количество иображений для обучения сети*/
        int samples = 60000;

        BufferedImage[] images = new BufferedImage[samples];

        /* Количество образцов для обучения */
        int[] digits = new int[samples];

        /* Массив изображения для обучения */
        File[] imagesFiles = new File("./train").listFiles();

        if (imagesFiles == null) {
            return;
        }

        if (imagesFiles.length == 0) {
            return;
        }

        /* Читаем изображения для тренировки сети */
        for (int i = 0; i < samples; i++) {
            /* Сохраняем индекс изображения в массиве */
            images[i] = ImageIO.read(imagesFiles[i]);

            /* Записываем в массив что за изображение должно быть на выходе (результат) */
            digits[i] = Integer.parseInt(imagesFiles[i].getName().charAt(10) + "");
        }

        /* Разбираем изображения 28х28 на пиксели вычисляя цвет пикселя*/
        double[][] inputs = new double[samples][784];

        for (int i = 0; i < samples; i++) {
            for (int x = 0; x < 28; x++) {
                for (int y = 0; y < 28; y++) {
                    inputs[i][x + y * 28] = (images[i].getRGB(x, y) & 0xff) / 255.0;
                }
            }
        }

        /* Количество проходов */
        int epochs = 1000;

        for (int i = 1; i < epochs; i++) {

            /* Процент правильности решения */
            int right = 0;

            /* Процент ошибок решения */
            double errorSum = 0;

            /* Количество повторов обучения */
            int batchSize = 100;

            for (int j = 0; j < batchSize; j++) {

                /* Берем случайное изображение */
                int imgIndex = (int) (Math.random() * samples);

                /* Определяем массив ответов */
                double[] targets = new double[10];

                /* Берем текущий правильный ответ по индексу на данное изображение */
                int digit = digits[imgIndex];

                /* Ставим текущий индекс в 1 */
                targets[digit] = 1;

                /* Вычисление текущего знаяения в сети */
                double[] outputs = neuralNetwork.feedForward(inputs[imgIndex]);

                /* Текущий ответ от сети */
                int maxDigit = 0;

                /* Текущий вес(процент) от сети */
                double maxDigitWeight = -1;

                for (int k = 0; k < 10; k++) {
                    /* Если вес больше -1 то берем это как ответ */
                    if (outputs[k] > maxDigitWeight) {
                        /* Присваиваем текущий вес */
                        maxDigitWeight = outputs[k];
                        /* Берем текущее число как ответ по текущему циклу */
                        maxDigit = k;
                    }
                }

                /* Сверяем текущий отчет с правильным результатом */
                if (digit == maxDigit) {
                    right++;
                }

                /* Текущий процент ошибки сети */
                for (int k = 0; k < 10; k++) {
                    errorSum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
                }

                /* Корректировка решения сети */
                neuralNetwork.backpropagation(targets);
            }
            System.out.println("epoch: " + i + ". correct: " + right + ". error: " + errorSum);
        }

        /* Запуск основной формы для проверки сети */
        new Thread(new FormDigits(neuralNetwork)).start();
    }
}
