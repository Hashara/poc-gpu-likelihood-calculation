#ifndef ALIGNMENT_IO_H
#define ALIGNMENT_IO_H

#include <string>
#include "alignment.h"

using namespace std;

/**
 * Reads a PHYLIP sequential DNA alignment file into an Alignment object.
 */
void readPhylipFile(const string& filename, Alignment& aln);

#endif // ALIGNMENT_IO_H
