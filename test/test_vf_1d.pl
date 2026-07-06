#!/usr/bin/env perl

# The test script is written in Perl. It runs the voigt_fit program with a test input file and checks the output file for the expected peak coordinates. 
# If the peak coordinates match the expected values, the test script returns a success code (0), otherwise it returns an error code (1). 
# The test script can be run using the command  perl test_dp.pl .
   


use strict;

`head -10 peaks.tab > peaks2.tab`;
my $exe = "./peak_fit_1d -method voigt -in test.ft1 -peak_in peaks2.tab -recon no -maxround 50";
my $out = `$exe`;



my @data1 = (8.3713,1.734975e+07);  
my @data2 = (8.3706,1.554571e+07);
my $data1_match = 0;
my $data2_match = 0;

open(IN, "fitted.tab" ) or die "Cannot open fitted.tab\n";

my $line;

while($line = <IN>)
{
    # skip if start with DATA, VARS or FORMAT
    if($line =~ /^DATA/ || $line =~ /^VARS/ || $line =~ /^FORMAT/)
    {
        next;
    }
    # get 4th and 5th columns as the peak coordinates in ppm
    $line=~s/^\s+//;
    my @fields = split(/\s+/, $line);

    #if it is very close to the expected value, we consider it match
    if(abs($fields[2] - $data1[0]) < 0.01 && abs($fields[4] - $data1[1]) < 1e5)
    {
        $data1_match = 1;
    }

    if(abs($fields[2] - $data2[0]) < 0.01 && abs($fields[4] - $data2[1]) < 1e5)
    {
        $data2_match = 1;
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
 