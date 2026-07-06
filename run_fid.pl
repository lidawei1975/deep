#!/usr/bin/perl
use strict;
use warnings;

sub run_fid_to_ft2_hsqc {
    my ($hsqc_acquisition_seq, $neg_imaginary, $execDir) = @_;

    my $command_step1 = $execDir . "deep-picker/release/fid -nus none -phase-in none -in hsqc_fid_file hsqc_acquisition_file hsqc_acquisition_file2 none";
    $command_step1 .= " -aqseq $hsqc_acquisition_seq";
    $command_step1 .= " -negative $neg_imaginary";
    $command_step1 .= " -first-only yes -di no -di-indirect no";
    $command_step1 .= " -out hsqc0.ft2 > hsqc_fid.log 2>&1";

    my $command_step2 = $execDir . "deep-picker/release/phase -real-only yes -in hsqc0.ft2 -out hsqc.ft2 -out-phase hsqc_phase.txt > hsqc_phase.log 2>&1";

    print("Running fid to ft2 conversion\n");
    system($command_step1);
    print("FID to FT2 conversion completed.\n");

    print("Running phase correction\n");
    system($command_step2);
    print("Phase correction completed.\n");

    my $command_fwhh_2d = $execDir . "deep-picker/release/fwhh_2d -in hsqc.ft2";
    print("Running peak width estimation\n");
    my @output = `$command_fwhh_2d`;
    print("FWHH 2D calculation completed.\n");
    chomp(@output);
    return 1 unless @output >= 2;

    my $second_last_line = $output[-2];
    my $last_line = $output[-1];

    my ($fwhh_direct) = split(/\s+/, $second_last_line);
    my ($fwhh_indirect) = split(/\s+/, $last_line);
    $fwhh_direct += 0;
    $fwhh_indirect += 0;

    print "FWHH Direct: $fwhh_direct\n";
    print "FWHH Indirect: $fwhh_indirect\n";

    my $zf_direct = 2;
    $zf_direct *= 2 while $fwhh_direct*$zf_direct < 5.0;

    my $zf_indirect = 2;
    $zf_indirect *= 2 while $fwhh_indirect*$zf_indirect < 5.0;

    my $phase_file = "hsqc_phase.txt";
    open(my $fh, '<', $phase_file) or return 1;
    my $line = <$fh>;
    close($fh);
    my @phase_parts = split(/\s+/, $line);
    my $phase_correction_indirect_p1 = $phase_parts[-1] + 0;

    my $apodization_indirect = "sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 0.5";
    if (abs($phase_correction_indirect_p1) > 30) {
        $apodization_indirect = "sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 1.0";
    }

    my $command = $execDir . "deep-picker/release/fid -nus none -di yes -di-indirect yes -phase-in hsqc_phase.txt";
    $command .= " -zf $zf_direct -zf-indirect $zf_indirect";
    $command .= " -in hsqc_fid_file hsqc_acquisition_file hsqc_acquisition_file2 none";
    $command .= " -out hsqc.ft2 -aqseq $hsqc_acquisition_seq -negative $neg_imaginary";
    $command .= " -apod-indirect '$apodization_indirect' -first-only no -out-json hsqc-fid-information.json";
    $command .= " > hsqc_fid2.log 2>&1";

    print("Running finla fid to ft2 conversion with proper peak widthes\n");
    system($command);
    print("Final FID to FT2 conversion completed.\n");

    return 0;
}

sub run_fid_to_ft2 {
    my ($file_name_prefix, $execDir) = @_;

    # Step 1: Convert FID to intermediate FT2
    my $command_step1 = "$execDir/deep-picker/release/fid -nus none -phase-in none -di no -di-indirect no " .
                        "-in ${file_name_prefix}_fid_file ${file_name_prefix}_acquisition_file ${file_name_prefix}_acquisition_file2 none " .
                        "-out ${file_name_prefix}0.ft2 > ${file_name_prefix}_fid.log 2>&1";
    print "Running ft2 conversion for ".$file_name_prefix."\n";
    system($command_step1);
    print "FID to intermediate FT2 conversion completed.\n";


    # Step 2: Phase correction
    my $command_step2 = "$execDir/deep-picker/release/phase -real-only yes -in ${file_name_prefix}0.ft2 -out none " .
                        "-out-phase ${file_name_prefix}_phase.txt > ${file_name_prefix}_phase.log 2>&1";
    print "Running phase correction for ".$file_name_prefix."\n";
    system($command_step2);
    print "Phase correction completed.\n";

    # Step 3: Read phase correction values
    my $phase_file = "${file_name_prefix}_phase.txt";
    open my $fh, '<',  $phase_file or return 1;
    my $line = <$fh>;
    close $fh;

    my @phase_values = split /\s+/, $line;
    my $phase_correction_indirect_p1 = $phase_values[-1];

    # Step 4: Determine apodization_indirect
    my $apodization_indirect = "sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 0.5";
    if (abs($phase_correction_indirect_p1) > 30) {
        $apodization_indirect = "sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 1.0";
    }

    # Step 5: Final FID to FT2 conversion using phase and apodization
    my $command_final = "$execDir/deep-picker/release/fid -nus none -di yes -di-indirect yes " .
                        "-phase-in ${file_name_prefix}_phase.txt " .
                        "-in ${file_name_prefix}_fid_file ${file_name_prefix}_acquisition_file ${file_name_prefix}_acquisition_file2 none " .
                        "-out ${file_name_prefix}.ft2 " .
                        "-apod-indirect \"$apodization_indirect\" > ${file_name_prefix}_fid2.log 2>&1";
    print "Running final ft2 conversion for ".$file_name_prefix."\n";
    system($command_final);
    print "Final FID to FT2 conversion completed.\n";

    return 0;
}

sub main {
    my $argc = @ARGV;
    my ($hsqc_acquisition_seq, $neg_imaginary, $execDir, $tocsy_exist, $hsqc_tocsy_exist) = @ARGV;


    run_fid_to_ft2_hsqc($hsqc_acquisition_seq, $neg_imaginary, $execDir);

    if ($tocsy_exist) {
        run_fid_to_ft2("tocsy", $execDir);
    }

    if ($hsqc_tocsy_exist) {
        run_fid_to_ft2("hsqc_tocsy", $execDir);
    }

    exit 0;
}

main();
