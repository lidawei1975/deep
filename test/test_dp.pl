#!/usr/bin/env perl

# The test script is written in Perl. It runs the deep_picker program with a test input file and checks the output file for the expected peak coordinates. If the peak coordinates match the expected values, the test script returns a success code (0), otherwise it returns an error code (1). The test script can be run using the command  perl test_dp.pl .
   



use strict;

my $exe = "./deep_picker -in plane001.ft2 -noise_level 233038.28125 -scale 18.5625 -scale2 11.1375 -out peaks.tab";

my $out = `$exe`;



my @data1 = (8.588399,123.804874);  
my @data2 = (8.601337,123.867233);
my $data1_match = 0;
my $data2_match = 0;

open(IN, "peaks.tab" ) or die "Cannot open peaks.tab\n";

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
    if(abs($fields[3] - $data1[0]) < 0.01 && abs($fields[4] - $data1[1]) < 0.1)
    {
        $data1_match = 1;
    }

    if(abs($fields[3] - $data2[0]) < 0.01 && abs($fields[4] - $data2[1]) < 0.1)
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
 
