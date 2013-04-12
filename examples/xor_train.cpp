/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2012 Steffen Nissen (sn@leenissen.dk)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>

#include "fann.h"



fann *createSpecificTrainedFann(unsigned int num_hiden_layers, unsigned int num_neurons_hidden_per_layer)
{
  if(num_hiden_layers > 7)
    return 0;
  const unsigned int num_input = 2;
  const unsigned int num_output = 1;
  const unsigned int num_layers = 2 + num_hiden_layers;
  const float desired_error = (const float) 0;
  const unsigned int max_epochs = 1000;
  const unsigned int epochs_between_reports = 0;

  struct fann *ann;
  struct fann_train_data *data;

  unsigned int p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;
  p1 = p2 = p3 = p4 = p5 = p6 = p7 = p8 = p9 = p10 = num_neurons_hidden_per_layer;
  switch (num_hiden_layers)
  {
  case 0:
    p1 = num_output;
    break;
  case 1:
    p2 = num_output;
    break;
  case 2:
    p3 = num_output;
    break;
  case 3:
    p4 = num_output;
    break;
  case 4:
    p5 = num_output;
    break;
  case 5:
    p6 = num_output;
    break;
  case 6:
    p7 = num_output;
    break;
  case 7:
    p8 = num_output;
    break;
  case 8:
    p9 = num_output;
    break;
  case 9:
    p10 = num_output;
    break;
  }

  ann = fann_create_standard(num_layers, num_input, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);

  data = fann_read_train_from_file("xor.data");

  fann_set_activation_steepness_hidden(ann, 1);
  fann_set_activation_steepness_output(ann, 1);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
  fann_set_bit_fail_limit(ann, 0.01f);

  fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

  fann_init_weights(ann, data);

  fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);

  fann_destroy_train(data);
  
  return ann;
}

int main()
{
  for(int i = 1; i < 7; ++i)
  {
    for(int j = 30; j < 512; j *=2)
    {
      fann *ann = createSpecificTrainedFann(i, 30);
      fann_destroy(ann);
    }    
  }

	return 0;
}
