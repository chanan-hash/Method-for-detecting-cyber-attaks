#include <iostream>
#include <fstream>
#include <string>

using namespace std;

double **readmat(string filename, int &size)
{
    ifstream file(filename); // Open file for reading

    if (!file)
    { // Check if file was opened successfully
        cout << "Error: Cannot open file " << filename << endl;
        exit(1);
    }

    // Read first line to determine matrix size
    string line;
    if (!getline(file, line))
    {
        cout << "Error: File is empty or unreadable." << endl;
        exit(1);
    }

    size = 0;
    for (char ch : line)
    {
        if (ch == ' ')
            size++;
    }
    size++; // Number of numbers in the first line determines the matrix size

    // Allocate memory for the matrix
    double **matrix = new double *[size];
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new double[size];
    }

    // Reset file pointer and read values into matrix
    file.clear();
    file.seekg(0, ios::beg);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (!(file >> matrix[i][j]))
            {
                cout << "Error: File contains invalid matrix format." << endl;
                exit(1);
            }
        }
    }

    file.close();
    return matrix;
}

bool palindrome(double **arr, int size, int col)
{

    if (col >= size)
    {
        cout << "Error: Column index out of bounds." << endl;
        exit(1);
    }
    for (int i = 0; i < size / 2; i++)
    {
        if (arr[i][col] != arr[size - 1 - i][col])
        {
            return false; // If mismatch found, it's not a palindrome
        }
    }
    return true; // If all elements match symmetrically, it's a palindrome
}


void printPalindromeCol(double **matrix, int size, int col)
{
    for (int i = 0; i < size; i++)
    {
        cout << matrix[i][col] << " ";
    }
    cout << endl;
}

double sum2x2(double **arr, int size, int row_index, int col_index)
{
    double sum = 0;
    for (int i = row_index; i < row_index + 2; i++)
    {
        for (int j = col_index; j < col_index + 2; j++)
        {
            sum += arr[i][j];
        }
    }
    return sum;
}

void printSubMatrix(double **matrix, int size, int row_index, int col_index)
{
    for (int i = row_index; i < row_index + 2; i++)
    {
        for (int j = col_index; j < col_index + 2; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// Function to print the matrix (for debugging)
void printMatrix(double **matrix, int size)
{
    if (!matrix)
        return;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// Function to free allocated memory
void freeMatrix(double **matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Main function to test reading a matrix from a file
int main()
{
    int size;
    string filename = "numbers.txt"; // Use your file name

    double **matrix = readmat(filename, size);

    if (matrix)
    {
        cout << "Matrix of size " << size << "x" << size << ":" << endl;
        printMatrix(matrix, size);
    }

    for (int i = 0; i < size; i++)
    {
        if (palindrome(matrix, size, i))
        {
            cout << "Column " << i << " is a palindrome." << endl;
            printPalindromeCol(matrix, size, i);
        }
    }

    cout << "The matrix has Palindromic columns." << endl;

    cout<< "Checking if the sum of 2x2 submatrices is equal." << endl;

    for (int i = 0; i <= size - 2; i++)
    {
        for (int j = 0; j <= size - 2; j++)
        {
            double sum1 = sum2x2(matrix, size, i, j);
            for (int k = i; k <= size - 2; k++)
            {
                for (int l = (k == i ? j + 1 : 0); l <= size - 2; l++)
                {
                    double sum2 = sum2x2(matrix, size, k, l);
                    if (sum1 == sum2)
                    {
                        std::cout << "Found two 2x2 submatrices with the same sum: " << sum1 << std::endl;
                        std::cout << "First submatrix starts at (" << i << ", " << j << ")" << std::endl;
                        std::cout << "Second submatrix starts at (" << k << ", " << l << ")" << std::endl;
                        std::cout << "First submatrix:" << std::endl;
                        printSubMatrix(matrix, size, i, j);
                        std::cout << "Second submatrix:" << std::endl;
                        printSubMatrix(matrix, size, k, l);
                    }
                }
            }
        }
    }



    freeMatrix(matrix, size);

    return 0;
}
