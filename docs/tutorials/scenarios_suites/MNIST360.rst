MNIST-360
-----------------

MNIST-360 is a scenario proposed in `"Dark Experience for General Continual Learning: a Strong, Simple Baseline" <https://arxiv.org/abs/2004.07211>`__

It is composed of 27 tasks with two MNIST classes each.
The two classes data have different rotation and the rotation change from one task to another.

.. image:: images/mnist360.png
  :width: 400
  :alt: Representation MNIST-360 from "Dark Experience for General Continual Learning: a Strong, Simple Baseline"

Continuum Scenarios
####

We do not propose a click and play scenario for MNIST, however, we show how to implement it with continuum.

.. code-block:: python

    from continuum.scenarios import ALMA
    scenario = ALMA(your_dataset, nb_megabatches=50)


    folder = "tests/samples/mnist360/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    continual_dataset = MNIST(data_path=DATA_PATH, download=False, train=True)

    for i in range(3):
        start_angle = 120 * i
        angle = int(120 / 9)

        list_rotation_first = [[transforms.RandomAffine(degrees=[start_angle + angle * j, start_angle + angle * j + 5])]
                               for j in range(10)]
        list_rotation_second = [
            [transforms.RandomAffine(degrees=[start_angle + 45 + angle * j, start_angle + 45 + angle * j + 5])] for j in
            range(10)]
        first_digit_scenario = ClassIncremental(continual_dataset, increment=1, transformations=list_rotation_first)
        second_digit_scenario = ClassIncremental(continual_dataset, increment=1, transformations=list_rotation_second)

        for task_id in range(9):
            taskset_1 = first_digit_scenario[task_id]
            taskset_2 = second_digit_scenario[1 + task_id % 9]

            # / ! \ we can not concatenate taskset here, since transformations would not work correctly

            loader_1 = DataLoader(taskset_1, batch_size=64)
            loader_2 = DataLoader(taskset_2, batch_size=64)

            nb_minibatches = min(len(loader_1), len(loader_2))
            for minibatch in range(nb_minibatches):
                x_1, y_1, t_1 = next(iter(loader_1))
                x_2, y_2, t_2 = next(iter(loader_2))

                x, y, t = torch.cat([x_1, x_2]), torch.cat([y_1, y_2]), torch.cat([t_1, t_2])

                # train here on x, y, t

                #### to visualize result ####
                # visualize_batch(batch=x[:100], number=100, shape=[28, 28, 1], path=folder + f"MNIST360_{task_id + 9 * i}.jpg")