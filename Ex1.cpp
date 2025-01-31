#include <iostream>

int* remove_duplicate(int arr[], int size, int* size2) {
  
	// Count occurrences of each element
	int* count = new int[size] {};
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (arr[i] == arr[j]) {
				count[i]++;
			}
		}
	}

	// Count unique elements that appear only once
	int new_size = 0;
	for (int i = 0; i < size; i++) {
		if (count[i] == 1) {
			new_size++;
		}
	}

	// Allocate exact size array
	int* new_arr = new int[new_size];
	int index = 0;
	for (int i = 0; i < size; i++) {
		if (count[i] == 1) {
			new_arr[index++] = arr[i];
		}
	}

	// Free temporary count array
	delete[] count;

	*size2 = new_size;
	return new_arr;
}

int main() {
	int arr[] = {1, 0, 3, 4, 7, 4, 0};
	int size = sizeof(arr) / sizeof(arr[0]);
	int new_size;

	int* new_arr = remove_duplicate(arr, size, &new_size);

	std::cout << "New array: ";
	for (int i = 0; i < new_size; i++) {
		std::cout << new_arr[i] << " ";
	}
	std::cout << std::endl;

    	std::cout << "New array size: " << new_size << std::endl;

	// Free allocated memory
	delete[] new_arr;

    
    //need to handle empty input
    
	int arr2[] = {};
	int size2 = sizeof(arr2) / sizeof(arr2[0]);
	int new_size2 = 0;
	int* new_arr2 = remove_duplicate(arr2, size2, &new_size);

	std::cout << "New array: ";
	for (int i = 0; i < new_size2; i++) {
		std::cout << new_arr2[i] << " ";
	}
	std::cout << std::endl;
	
	delete[] new_arr2;

	return 0;
}