#!/usr/bin/env perl

# The test script is written in Perl. It runs the fid program and phase program with a test input file and checks the output file for the expected phase correction values 
   



use strict;

my $exe1 = "./fid -process full  -nus none -out test.ft2 -di no -di-indirect no -phase-in none";
my $exe2 = "./phase -in test.ft2 -real-only yes -user no -user-indirect no";

my $out1 = `$exe1`;
my $out2 = `$exe2`;



my @expected = (46,20,90,0);


open(IN, "phase-correction.txt" ) or die "Cannot open phase-correction.txt\n";

my $line;
my $data1_match = 0;
my $data2_match = 0;

$line = <IN>;

    
$line=~s/^\s+//;
my @fields = split(/\s+/, $line);

#if it is very close to the expected value, we consider it match
if(abs($fields[0] - $expected[0]) < 10  && abs($fields[1] - $expected[1]) < 10)
{
    $data1_match = 1;
}

if(abs($fields[2] - $expected[2]) < 0.01 && abs($fields[3] - $expected[3]) < 0.1)
{
    $data2_match = 1;
}
close(IN);



my $return_code = 1; # code 1 means error, 0 means success

if($data1_match == 1 && $data2_match == 1)
{
    $return_code = 0;
}

exit($return_code);
 
