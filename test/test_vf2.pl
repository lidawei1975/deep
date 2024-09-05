#!/usr/bin/env perl

# The test script is written in Perl. It runs the voigt_fit program with 3 planes for pseudo-3D fitting and checks the output file for the expected peak coordinates. 
# If the peak coordinates match the expected values, the test script returns a success code (0), otherwise it returns an error code (1). 
# The test script can be run using the command  perl test_dp.pl .
   


use strict;

my $exe = "./voigt_fit -method voigt -in plane001.ft2 plane002.ft2 plane003.ft2 -peak_in fitted.tab -out fitted2.tab -recon no -noise_level 233038.28125 -scale 18.5625 -scale2 11.1375";

my $out = `$exe`;

print($out);



my @data1 = (8.585261,123.779270,0.7085,0.4992);  
my @data2 = (8.598069,123.841416,0.6532,0.4401);
my $data1_match = 0;
my $data2_match = 0;

open(IN, "fitted2.tab" ) or die "Cannot open fitted.tab\n";

my $line;

while($line = <IN>)
{
    # skip if start with DATA, VARS or FORMAT
    if($line =~ /^DATA/ || $line =~ /^VARS/ || $line =~ /^FORMAT/)
    {
        next;
    }
    # get 4th and 5th columns as the peak coordinates in ppm
    # and 24th and 25th columns as the relative peak intensities
    $line=~s/^\s+//;
    my @fields = split(/\s+/, $line);

    print("fields[3]: $fields[3], fields[4]: $fields[4], fields[24]: $fields[24], fields[25]: $fields[25]\n");

    #if it is very close to the expected value, we consider it match
    if(abs($fields[3] - $data1[0]) < 0.01 && abs($fields[4] - $data1[1]) < 0.1 && abs($fields[24] - $data1[2]) < 0.01 && abs($fields[25] - $data1[3]) < 0.01)
    {
        $data1_match += 1;
    }

    if(abs($fields[3] - $data2[0]) < 0.01 && abs($fields[4] - $data2[1]) < 0.1 && abs($fields[24] - $data2[2]) < 0.01 && abs($fields[25] - $data2[3]) < 0.01)
    {
        $data2_match += 1;
    }
}

close(IN);

print("data1_match: $data1_match\n");
print("data2_match: $data2_match\n");

my $return_code = 1; # code 1 means error, 0 means success

if($data1_match == 1 && $data2_match == 1)
{
    $return_code = 0;
}

exit($return_code);
 
